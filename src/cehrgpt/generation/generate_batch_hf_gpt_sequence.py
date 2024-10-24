import datetime
import os
import random
import uuid
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import GenerationConfig
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.cehrgpt_args import create_inference_base_arg_parser
from cehrgpt.gpt_utils import get_cehrgpt_output_folder
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import NA, CehrGptTokenizer

LOG = logging.get_logger("transformers")


def generate_single_batch(
    model,
    tokenizer,
    batch_size,
    demographic_info,
    max_new_tokens=512,
    mini_num_of_concepts=1,
    top_p=0.95,
    top_k=50,
    temperature=1.0,
    repetition_penalty=1.0,
    num_beams=1,
    num_beam_groups=1,
    epsilon_cutoff=0.0,
    device: Any = "cpu",
) -> Dict[str, Any]:
    random_prompts = random.sample(demographic_info, batch_size)

    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=repetition_penalty,
            max_length=max_new_tokens,
            min_length=mini_num_of_concepts,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            bos_token_id=tokenizer.end_token_id,
            eos_token_id=tokenizer.end_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
            renormalize_logits=True,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            epsilon_cutoff=epsilon_cutoff,
        )
        batched_prompts = torch.tensor(random_prompts).to(device)
        results = model.generate(
            inputs=batched_prompts, generation_config=generation_config
        )

    sequences = [tokenizer.decode(seq.cpu().numpy()) for seq in results.sequences]
    if results.sequence_val_masks is not None:
        value_indicators = [
            m[: len(s)]
            for m, s in zip(
                results.sequence_val_masks.detach().cpu().numpy(),
                sequences,
            )
        ]
    else:
        value_indicators = [None] * len(sequences)
    if results.sequence_vals is not None:
        values = [
            v[: len(s)]
            for v, s in zip(
                results.sequence_vals.detach().to(torch.float32).cpu().numpy(),
                sequences,
            )
        ]
    else:
        values = [None] * len(sequences)
    return {
        "sequences": sequences,
        "value_indicators": value_indicators,
        "values": values,
    }


def normalize_value(
    seq: Sequence[str],
    value_indicators: Sequence[bool],
    values: Sequence[float],
    tokenizer: CehrGptTokenizer,
) -> Tuple[Optional[Sequence[float]], Optional[Sequence[str]]]:
    if value_indicators is not None and values is not None:
        normalized_values = []
        units = []
        for concept_id, value_indicator, value in zip(seq, value_indicators, values):
            if value_indicator:
                normalized_value, unit = tokenizer.denormalize(concept_id, value)
                normalized_values.append(normalized_value)
                units.append(unit)
            else:
                normalized_values.append(0.0)
                units.append(NA)
        return normalized_values, units
    return None, None


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_folder)
    cehrgpt_model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            args.model_folder,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
            torch_dtype=(
                torch.bfloat16 if is_flash_attn_2_available() else torch.float32
            ),
        )
        .eval()
        .to(device)
    )
    cehrgpt_model.generation_config.pad_token_id = cehrgpt_tokenizer.pad_token_id
    cehrgpt_model.generation_config.eos_token_id = cehrgpt_tokenizer.end_token_id
    cehrgpt_model.generation_config.bos_token_id = cehrgpt_tokenizer.end_token_id

    folder_name = get_cehrgpt_output_folder(args, cehrgpt_tokenizer)
    output_folder_name = os.path.join(
        args.output_folder, folder_name, "generated_sequences"
    )

    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    LOG.info(f"Loading tokenizer at {args.model_folder}")
    LOG.info(f"Loading model at {args.model_folder}")
    LOG.info(f"Write sequences to {output_folder_name}")
    LOG.info(f"Context window {args.context_window}")
    LOG.info(f"Temperature {args.temperature}")
    LOG.info(f"Repetition Penalty {args.repetition_penalty}")
    LOG.info(f"Sampling Strategy {args.sampling_strategy}")
    LOG.info(f"Num beam {args.num_beams}")
    LOG.info(f"Num beam groups {args.num_beam_groups}")
    LOG.info(f"Epsilon cutoff {args.epsilon_cutoff}")
    LOG.info(f"Top P {args.top_p}")
    LOG.info(f"Top K {args.top_k}")
    LOG.info(f"Loading demographic_info at {args.demographic_data_path}")

    data = pd.read_parquet(args.demographic_data_path)

    data = pd.read_parquet(args.demographic_data_path)
    # data = data[data.num_of_concepts >= args.min_num_of_concepts]
    demographic_info = data.concept_ids.apply(lambda concept_list: concept_list[0:4])
    demographic_info = [cehrgpt_tokenizer.encode(_) for _ in demographic_info]

    num_of_batches = args.num_of_patients // args.batch_size + 1
    sequence_to_flush = []
    current_person_id = 1
    for i in range(num_of_batches):
        print(f"{datetime.datetime.now()}: Batch {i} started")
        batch_sequences = generate_single_batch(
            cehrgpt_model,
            cehrgpt_tokenizer,
            args.batch_size,
            demographic_info,
            max_new_tokens=args.context_window,
            mini_num_of_concepts=args.min_num_of_concepts,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            epsilon_cutoff=args.epsilon_cutoff,
            device=device,
        )

        # Clear the cache
        torch.cuda.empty_cache()

        for seq, value_indicator, value in zip(
            batch_sequences["sequences"],
            batch_sequences["value_indicators"],
            batch_sequences["values"],
        ):
            normalized_values, units = normalize_value(
                seq, value_indicator, value, cehrgpt_tokenizer
            )
            output = {"concept_ids": seq, "person_id": current_person_id}
            if normalized_values is not None:
                output["concept_values"] = normalized_values
            if value_indicator is not None:
                output["concept_value_masks"] = value_indicator
            if units is not None:
                output["units"] = units

            sequence_to_flush.append(output)
            current_person_id += 1

        if len(sequence_to_flush) >= args.buffer_size:
            print(f"{datetime.datetime.now()}: Flushing to the Disk at Batch {i}")
            pd.DataFrame(
                sequence_to_flush,
                columns=[
                    "concept_ids",
                    "person_id",
                    "concept_values",
                    "concept_value_masks",
                    "units",
                ],
            ).to_parquet(os.path.join(output_folder_name, f"{uuid.uuid4()}.parquet"))
            sequence_to_flush.clear()

    if len(sequence_to_flush) > 0:
        print(f"{datetime.datetime.now()}: Flushing to the Disk at Final Batch")
        pd.DataFrame(
            sequence_to_flush,
            columns=[
                "concept_ids",
                "person_id",
                "concept_values",
                "concept_value_masks",
                "units",
            ],
        ).to_parquet(os.path.join(output_folder_name, f"{uuid.uuid4()}-last.parquet"))


def create_arg_parser():
    base_arg_parser = create_inference_base_arg_parser(
        description="Arguments for generating patient sequences"
    )
    base_arg_parser.add_argument(
        "--num_of_patients",
        dest="num_of_patients",
        action="store",
        type=int,
        help="The number of patients that will be generated",
        required=True,
    )

    base_arg_parser.add_argument(
        "--demographic_data_path",
        dest="demographic_data_path",
        action="store",
        help="The path for your concept_path",
        required=True,
    )
    return base_arg_parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
