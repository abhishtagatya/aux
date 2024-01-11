import argparse
import os

from datasets import load_dataset, load_metric

if __name__ == '__main__':
    # Get Script Argument
    parser = argparse.ArgumentParser(description="Wav2Vec2.0 ASR Fine-Tune")
    parser.add_argument("ft_dataset", help="Fine-Tuning Dataset from HF", type=str)
    args = parser.parse_args()

    # Import Dataset
    ds = load_dataset(args.ft_dataset, data_dir="./data/timit_asr")
    print(ds)