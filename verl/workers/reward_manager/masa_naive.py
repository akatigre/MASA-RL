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

from collections import defaultdict
import re
import torch
import json
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.utils import extract_last_json_object
from verl.workers.reward_manager import register


@register("masa_naive")
class MasaRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the MasaRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        
        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        contain_uid = data[0].non_tensor_batch.get("uid", None)
        stats_for_meta_cog = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            if contain_uid:
                uid = data_item.non_tensor_batch["uid"]
                if uid.endswith("_meta_cog"):
                    continue

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum(
            )
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum(
            )
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if contain_uid:
                stats_for_meta_cog[uid].append(
                    {"response": response_str, "score": score, "length": valid_response_length.item()})

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        all_uid_meta_info = None
        if contain_uid:
            all_uid_meta_info = defaultdict(list)
            already_print_data_sources = {}
            for i, data_item in enumerate(data):
                
                uid = data_item.non_tensor_batch['uid']
                if not uid.endswith("_meta_cog"):
                    continue  # skip non-meta-cog data
                
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                uid = uid.replace("_meta_cog", "")
                stats = stats_for_meta_cog[uid]
                gt_stats = {key: [stat[key] for stat in stats]
                            for key in stats[0].keys()}

                meta_prompt_ids = data_item.batch['prompts']
                prompt_length = meta_prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum(
                )
                valid_prompt_ids = meta_prompt_ids[-valid_prompt_length:]
                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum(
                )
                valid_response_ids = response_ids[:valid_response_length]

                meta_prompt_str = self.tokenizer.decode(
                    valid_prompt_ids, skip_special_tokens=True)
                meta_response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=True)
                gt_stats["problem"] = meta_prompt_str
                meta_json_dict = extract_last_json_object(meta_response_str)
                match = re.search(r"<meta>(.*?)</meta>", meta_response_str, re.DOTALL)
                if match:
                    meta_reasoning = match.group(1).strip()
                else:
                    meta_reasoning = ""
                gt_stats["reasoning"] = meta_reasoning
                    
                meta_score = self.compute_score(
                    data_source="meta_cog",
                    solution_str=meta_json_dict,
                    ground_truth=gt_stats,
                    extra_info=extra_info
                )
                # Change the predicted value of length and pass_rate to ground truth
                correct_length = sum([l for l, correct in zip(gt_stats["length"], gt_stats["score"]) if correct == 1]) / len(gt_stats["length"])
                correct_pass_rate = sum(gt_stats["score"]) / len(gt_stats["score"])
                meta_json_dict["solution_length"] = correct_length
                meta_json_dict["pass_rate"] = correct_pass_rate
                per_uid_meta_info = {
                    'prompt': meta_prompt_str,
                    'response': meta_response_str,
                    'meta_json': meta_json_dict,
                    'meta_reasoning': meta_reasoning,
                    'notion_score': meta_score["notion_score"]
                }
                all_uid_meta_info[uid].append(per_uid_meta_info)

                if isinstance(meta_score, dict):
                    reward = meta_score["score"]
                    # Store the information including original reward
                    for key, value in meta_score.items():
                        reward_extra_info["meta_cog_" + key].append(value)
                else:
                    reward = meta_score
                reward_tensor[i, valid_response_length - 1] = reward

                if "meta_cog" not in already_print_data_sources:
                    already_print_data_sources["meta_cog"] = 0

                if already_print_data_sources["meta_cog"] < self.num_examine:
                    already_print_data_sources["meta_cog"] += 1
                    print("[META prompt]", meta_prompt_str)
                    print("[META response]", meta_response_str)
                    print("[META reasoning]", meta_reasoning)
                    print("[META corrected json]", meta_json_dict)
                    if isinstance(meta_score, dict):
                        for key, value in meta_score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", meta_score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "all_uid_meta_info": all_uid_meta_info,
            }
        else:
            return reward_tensor
