# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the DAPO dataset to parquet format
"""
import pandas as pd
import argparse
import json
import os

import datasets

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/deepscaler")
    parser.add_argument("--dataset_name", default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()
    
    print(f"Loading the {args.dataset_name} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(args.dataset_name)['train']
    
    def map_fn(split):
        def process_fn(example, idx):
            instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
            question = example.pop("problem")
            question = question + " " + instruction_following
            solution = {
                "question": question,
                "answer": example["answer"],
                "target": example["answer"],
            }
            data = {
                "data_source": args.dataset_name,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    "split": split,
                    "index": idx
                }
            }
            return data
        return process_fn
    train_dataset = dataset.map(function=map_fn("train"), with_indices=True)
    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir)
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    example = train_dataset[0]
    with open(os.path.join(args.local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)