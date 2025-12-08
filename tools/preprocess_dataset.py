"""Preprocess an SFT dataset into the native RLKit format."""

import argparse
import logging
from datasets import load_dataset, get_dataset_config_names, Dataset
from transformers import AutoTokenizer
from rlkit.config.sft import DatasetType
from rlkit.data.sft_datasets import transform_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("dataset_type", type=DatasetType)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--tokenizer-name", type=str, required=False, default=None)
    parser.add_argument("--ds-config", type=str, required=False, default=None)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-private", action="store_true")
    args = parser.parse_args()

    # Load tokenizer
    if args.tokenizer_name is None:
        logging.info("--tokenizer-name not provided, non-pretokenized datasets will crash.")
        tokenizer = None
    else:
        logging.info(f"Loading tokenizer '{args.tokenizer_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    ds_configs = get_dataset_config_names(args.dataset_name)
    if len(ds_configs) > 1:
        if args.ds_config is None:
            raise ValueError(f"Dataset has configs {ds_configs}, but --ds-config was not provided.")
        else:
            dataset = load_dataset(args.dataset_name, args.ds_config)
    else:
        dataset = load_dataset(args.dataset_name)

    assert isinstance(dataset, Dataset), f"Expected Dataset, got {type(dataset)}"

    logging.info("Loaded dataset. Applying transformation...")

    dataset = transform_dataset(dataset, args.dataset_type, tokenizer, num_proc=args.num_proc)

    if not args.push_to_hub:
        logging.info(f"Saving dataset to '{args.output_path}'")
        dataset.save_to_disk(args.output_path)
    else:
        logging.info(f"Pushing dataset to Hugging Face repo '{args.output_path}'...")
        dataset.push_to_hub(args.output_path, private=args.hf_private, num_proc=args.num_proc)
