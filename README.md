# CEHRGPT

![Python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/knatarajan-lab/cehrgpt/blob/main/LICENSE)
[![contributors](https://img.shields.io/github/contributors/knatarajan-lab/cehrgpt.svg)](https://github.com/knatarajan-lab/cehrgpt/graphs/contributors)

## Description
CEHRGPT is a synthetic data generation model developed to handle structured electronic health records (EHR) with enhanced privacy and reliability. It leverages state-of-the-art natural language processing techniques to create realistic, anonymized patient data that can be used for research and development without compromising patient privacy.

## Features
- **Synthetic Patient Data Generation**: Generates comprehensive patient profiles including demographics, medical history, treatment courses, and outcomes.
- **Privacy-Preserving**: Implements techniques to ensure the generated data does not reveal identifiable information.
- **Compatibility with OMOP**: Fully compatible with the OMOP common data model, allowing seamless integration with existing healthcare data systems.
- **Extensible**: Designed to be adaptable to new datasets and different EHR systems.

## Installation
To install CEHRGPT, clone this repository and install the required dependencies.

```bash
git clone https://github.com/knatarajan-lab/cehrgpt.git
cd cehrgpt
pip install .
```

## Pretrain
Pretrain cehrgpt using the Hugging Face trainer, the parameters can be found in the sample configuration yaml
```bash
mkdir test_results
# This is NOT required when streaming is set to true
mkdir test_dataset_prepared
python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner sample_configs/cehrgpt_pretrain_sample_config.yaml
```

## Citation
```
@article{cehrgpt2024,
  title={CEHRGPT: Synthetic Data Generation for Electronic Health Records},
  author={Natarajan, K and others},
  journal={arXiv preprint arXiv:2402.04400},
  year={2024}
}
```
