import os
from typing import Optional, Union

from cehrbert.data_generators.hf_data_generator.meds_utils import (
    create_dataset_from_meds_reader,
)
from cehrbert.runners.hf_runner_argument_dataclass import (
    DataTrainingArguments,
    ModelArguments,
)
from cehrbert.runners.runner_util import (
    generate_prepared_ds_path,
    get_last_hf_checkpoint,
    get_meds_extension_path,
    load_parquet_as_dataset,
    parse_runner_args,
)
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_from_disk
from transformers import AutoConfig, Trainer, set_seed
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_pretraining_dataset
from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

LOG = logging.get_logger("transformers")


def tokenizer_exists(model_args: ModelArguments) -> bool:
    # Try to load the pretrained tokenizer
    try:
        CehrGptTokenizer.from_pretrained(
            os.path.abspath(model_args.tokenizer_name_or_path)
        )
        return True
    except Exception:
        LOG.info(f"The tokenizer does not exist at {model_args.tokenizer_name_or_path}")
        return False


def load_and_create_tokenizer(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    dataset: Optional[Union[Dataset, DatasetDict]] = None,
) -> CehrGptTokenizer:
    # Try to load the pretrained tokenizer
    tokenizer_abspath = os.path.abspath(model_args.tokenizer_name_or_path)
    try:
        tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_abspath)
    except Exception as e:
        LOG.warning(e)
        if dataset is None:
            raise RuntimeError(
                f"Failed to load the tokenizer from {tokenizer_abspath} with the error \n{e}\n"
                f"Tried to create the tokenizer, however the dataset is not provided."
            )

        tokenizer = CehrGptTokenizer.train_tokenizer(
            dataset, ["concept_ids"], {}, data_args
        )
        tokenizer.save_pretrained(tokenizer_abspath)

    return tokenizer


def load_and_create_model(
    model_args: ModelArguments, tokenizer: CehrGptTokenizer
) -> CEHRGPT2LMHeadModel:
    attn_implementation = (
        "flash_attention_2" if is_flash_attn_2_available() else "eager"
    )
    try:
        model_abspath = os.path.abspath(model_args.model_name_or_path)
        model_config = AutoConfig.from_pretrained(
            model_abspath, attn_implementation=attn_implementation
        )
    except Exception as e:
        LOG.warning(e)
        model_config = CEHRGPTConfig(
            vocab_size=tokenizer.vocab_size,
            time_token_vocab_size=tokenizer.time_token_vocab_size,
            bos_token_id=tokenizer.end_token_id,
            eos_token_id=tokenizer.end_token_id,
            lab_token_ids=tokenizer.lab_token_ids,
            token_to_time_token_mapping=tokenizer.token_to_time_token_mapping,
            attn_implementation=attn_implementation,
            **model_args.as_dict(),
        )
    return CEHRGPT2LMHeadModel(model_config)


def main():
    data_args, model_args, training_args = parse_runner_args()

    if data_args.streaming:
        # This is for disabling the warning message https://github.com/huggingface/transformers/issues/5486
        # This happens only when streaming is enabled
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # The iterable dataset doesn't have sharding implemented, so the number of works has to be set to 0
        # Otherwise the trainer will throw an error
        training_args.dataloader_num_workers = 0
        training_args.dataloader_prefetch_factor = 0

    prepared_ds_path = generate_prepared_ds_path(data_args, model_args)
    if os.path.exists(os.path.join(data_args.data_folder, "dataset_dict.json")):
        LOG.info(f"Loading prepared dataset from disk at {data_args.data_folder}...")
        processed_dataset = load_from_disk(data_args.data_folder)
        # If the data has been processed in the past, it's assume the tokenizer has been created before.
        # we load the CEHR-GPT tokenizer from the output folder, otherwise an exception will be raised.
        if not tokenizer_exists(model_args):
            raise RuntimeError(
                f"The dataset has been tokenized but the corresponding tokenizer: "
                f"{model_args.tokenizer_name_or_path} does not exist"
            )
        cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path
        )
    elif any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
        # If the data has been processed in the past, it's assume the tokenizer has been created before.
        # we load the CEHR-GPT tokenizer from the output folder, otherwise an exception will be raised.
        if not tokenizer_exists(model_args):
            raise RuntimeError(
                f"The dataset has been tokenized but the corresponding tokenizer: "
                f"{model_args.tokenizer_name_or_path} does not exist"
            )
        cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path
        )
    else:
        # If the data is in the MEDS format, we need to convert it to the CEHR-BERT format
        if data_args.is_data_in_med:
            meds_extension_path = get_meds_extension_path(
                data_folder=data_args.data_folder,
                dataset_prepared_path=data_args.dataset_prepared_path,
            )
            try:
                LOG.info(
                    "Trying to load the MEDS extension from disk at %s...",
                    meds_extension_path,
                )
                dataset = load_from_disk(meds_extension_path)
                if data_args.streaming:
                    if isinstance(dataset, DatasetDict):
                        dataset = {
                            k: v.to_iterable_dataset(
                                num_shards=training_args.dataloader_num_workers
                            )
                            for k, v in dataset.items()
                        }
                    else:
                        dataset = dataset.to_iterable_dataset(
                            num_shards=training_args.dataloader_num_workers
                        )
            except FileNotFoundError as e:
                LOG.exception(e)
                dataset = create_dataset_from_meds_reader(
                    data_args, is_pretraining=True
                )
                if not data_args.streaming:
                    dataset.save_to_disk(meds_extension_path)
        else:
            # Load the dataset from the parquet files
            dataset = load_parquet_as_dataset(
                data_args.data_folder, split="train", streaming=data_args.streaming
            )
            # If streaming is enabled, we need to manually split the data into train/val
            if data_args.streaming and data_args.validation_split_num:
                dataset = dataset.shuffle(buffer_size=10_000, seed=training_args.seed)
                train_set = dataset.skip(data_args.validation_split_num)
                val_set = dataset.take(data_args.validation_split_num)
                dataset = DatasetDict({"train": train_set, "test": val_set})
            elif data_args.validation_split_percentage:
                dataset = dataset.train_test_split(
                    test_size=data_args.validation_split_percentage,
                    seed=training_args.seed,
                )
            else:
                raise RuntimeError(
                    f"Can not split the data. If streaming is enabled, validation_split_num needs to be "
                    f"defined, otherwise validation_split_percentage needs to be provided. "
                    f"The current values are:\n"
                    f"validation_split_percentage: {data_args.validation_split_percentage}\n"
                    f"validation_split_num: {data_args.validation_split_num}\n"
                    f"streaming: {data_args.streaming}"
                )

        # Create the CEHR-GPT tokenizer if it's not available in the output folder
        cehrgpt_tokenizer = load_and_create_tokenizer(
            data_args=data_args, model_args=model_args, dataset=dataset
        )
        # sort the patient features chronologically and tokenize the data
        processed_dataset = create_cehrgpt_pretraining_dataset(
            dataset=dataset, cehrgpt_tokenizer=cehrgpt_tokenizer, data_args=data_args
        )
        # only save the data to the disk if it is not streaming
        if not data_args.streaming:
            processed_dataset.save_to_disk(prepared_ds_path)

    def filter_func(examples):
        return [_ >= data_args.min_num_tokens for _ in examples["num_of_concepts"]]

    # Create the args for batched filtering
    filter_args = {"batched": True, "batch_size": data_args.preprocessing_batch_size}
    # If the dataset is not in a streaming mode, we could add num_proc to enable parallelization
    if not data_args.streaming:
        filter_args["num_proc"] = data_args.preprocessing_num_workers

    # The filter can't be applied to a DatasetDict of IterableDataset (in case of streaming)
    # we need to iterate through all the datasets and apply the filter separately
    if isinstance(processed_dataset, DatasetDict) or isinstance(
        processed_dataset, IterableDatasetDict
    ):
        for key in processed_dataset.keys():
            processed_dataset[key] = processed_dataset[key].filter(
                filter_func, **filter_args
            )
    else:
        processed_dataset = processed_dataset.filter(filter_func, **filter_args)

    model = load_and_create_model(model_args, cehrgpt_tokenizer)

    # Detecting last checkpoint.
    last_checkpoint = get_last_hf_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not data_args.streaming:
        processed_dataset.set_format("pt")

    eval_dataset = None
    if isinstance(processed_dataset, DatasetDict):
        train_dataset = processed_dataset["train"]
        if "test" in processed_dataset:
            eval_dataset = processed_dataset["test"]
    else:
        train_dataset = processed_dataset

    trainer = Trainer(
        model=model,
        data_collator=CehrGptDataCollator(
            tokenizer=cehrgpt_tokenizer,
            max_length=model_args.max_position_embeddings,
            shuffle_records=data_args.shuffle_records,
            include_values=model_args.include_values,
            include_ttv_prediction=model_args.include_ttv_prediction,
            use_sub_time_tokenization=model_args.use_sub_time_tokenization,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
