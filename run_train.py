#!/usr/bin/env python
import argparse
from src.training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)