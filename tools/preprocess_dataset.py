import argparse
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from rlkit.data.datasets import transform_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("dataset_type", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--tokenizer-name", type=str, required=False, default=None)
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
    
    dataset = load_dataset(args.dataset_name)
    logging.info(f"Loaded dataset with splits {list(dataset.keys())}. Applying transformation...")
    
    for split in dataset.keys():
        logging.info(f"Transforming split '{split}'")
        dataset[split] = transform_dataset(dataset[split], args.dataset_type, tokenizer, num_proc=args.num_proc)
    
    if not args.push_to_hub:
        logging.info(f"Saving dataset to '{args.output_path}'")
        dataset.save_to_disk(args.output_path)
    else:
        logging.info(f"Pushing dataset to Hugging Face repo '{args.output_path}'...")
        dataset.push_to_hub(args.output_path, private=args.hf_private, num_proc=args.num_proc)
    