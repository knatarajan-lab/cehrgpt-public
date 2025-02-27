import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from datasets import disable_caching

from cehrgpt.generation.generate_batch_hf_gpt_sequence import create_arg_parser
from cehrgpt.generation.generate_batch_hf_gpt_sequence import main as generate_main
from cehrgpt.models.pretrained_embeddings import (
    PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
    PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
)
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import main as train_main

disable_caching()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"


class HfCehrGptRunnerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get the root folder of the project
        root_folder = Path(os.path.abspath(__file__)).parent.parent.parent.parent
        cls.data_folder = os.path.join(root_folder, "sample_data", "pretrain")
        cls.pretrained_embedding_folder = os.path.join(
            root_folder, "sample_data", "pretrained_embeddings"
        )
        # Create a temporary directory to store model and tokenizer
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_folder_path = os.path.join(cls.temp_dir, "model")
        Path(cls.model_folder_path).mkdir(parents=True, exist_ok=True)
        cls.dataset_prepared_path = os.path.join(cls.temp_dir, "dataset_prepared_path")
        Path(cls.dataset_prepared_path).mkdir(parents=True, exist_ok=True)
        cls.generation_folder_path = os.path.join(cls.temp_dir, "generation")
        Path(cls.generation_folder_path).mkdir(parents=True, exist_ok=True)
        for file_name in [
            PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME,
            PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
        ]:
            shutil.copy(
                os.path.join(cls.pretrained_embedding_folder, file_name),
                os.path.join(cls.model_folder_path, file_name),
            )

    @classmethod
    def tearDownClass(cls):
        # Remove the temporary directory
        shutil.rmtree(cls.temp_dir)

    def test_1_train_model(self):
        sys.argv = [
            "hf_cehrgpt_pretraining_runner.py",
            "--model_name_or_path",
            self.model_folder_path,
            "--tokenizer_name_or_path",
            self.model_folder_path,
            "--output_dir",
            self.model_folder_path,
            "--data_folder",
            self.data_folder,
            "--dataset_prepared_path",
            self.dataset_prepared_path,
            "--pretrained_embedding_path",
            self.model_folder_path,
            "--max_steps",
            "100",
            "--save_steps",
            "100",
            "--save_strategy",
            "steps",
            "--hidden_size",
            "192",
            "--use_sub_time_tokenization",
            "false",
            "--include_ttv_prediction",
            "false",
            "--include_values",
            "true",
            "--lab_token_penalty",
            "true",
            "--entropy_penalty",
            "true",
            "--validation_split_num",
            "10",
            "--streaming",
            "true",
        ]
        train_main()
        # Teacher force the prompt to consist of [year][age][gender][race][VS] then inject the random vector before [VS]

    def test_2_generate_model(self):
        sys.argv = [
            "generate_batch_hf_gpt_sequence.py",
            "--model_folder",
            self.model_folder_path,
            "--tokenizer_folder",
            self.model_folder_path,
            "--output_folder",
            self.generation_folder_path,
            "--context_window",
            "128",
            "--num_of_patients",
            "16",
            "--batch_size",
            "4",
            "--buffer_size",
            "16",
            "--sampling_strategy",
            "TopPStrategy",
            "--demographic_data_path",
            self.data_folder,
        ]
        args = create_arg_parser().parse_args()
        generate_main(args)
        generated_sequences = pd.read_parquet(self.generation_folder_path)
        for concept_ids in generated_sequences.concept_ids:
            print(concept_ids)


if __name__ == "__main__":
    unittest.main()
