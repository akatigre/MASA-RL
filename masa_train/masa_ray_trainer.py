# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    Role,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_response_mask,
    compute_advantage,
    RayPPOTrainer
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
import verl.utils.torch_functional as verl_F
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = type[Worker]


class RayMASATrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn, train_dataset, val_dataset, collate_fn, train_sampler, device_name)
        self.is_meta_cog = config.data.get('meta_cog_data', False)
        
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps,
                            initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            if self.config.actor_rollout_ref.get("meta_sft"):
                prompts = []
                responses = []
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.start_profile(
                            role="e2e", profile_step=self.global_steps)
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                # pop those keys for generation
                batch_keys_to_pop = ['input_ids', 'attention_mask', 'position_ids']
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                if "extra_info" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("extra_info")

                response_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop, # ['raw_prompt_ids', 'raw_prompt', 'tools_kwargs', 'interaction_kwargs', 'index', 'extra_info']
                )

                # pass global_steps to trace
                response_batch.meta_info["global_steps"] = self.global_steps

                if self.is_meta_cog:
                    meta_tensor_keys = ["input_ids", "attention_mask", "position_ids"]
                    meta_non_tensor_keys = ["raw_prompt_ids"] 
                    if "raw_prompt" in batch.non_tensor_batch:
                        meta_non_tensor_keys += ["raw_prompt"]
                    
                    meta_batch = batch.pop(
                        batch_keys = ["meta_cog_" + key for key in meta_tensor_keys],
                        non_tensor_batch_keys = ["meta_cog_" + key for key in meta_non_tensor_keys],
                    )
                    meta_batch = DataProto.from_dict(
                        tensors = {key: meta_batch.batch[f"meta_cog_{key}"] for key in meta_tensor_keys},
                        non_tensors = {key: meta_batch.non_tensor_batch[f"meta_cog_{key}"] for key in meta_non_tensor_keys},
                    )
                    for k, v in response_batch.non_tensor_batch.items():
                        if k not in meta_batch.non_tensor_batch:
                            meta_batch.non_tensor_batch[k] = v # Update non_tensors with keys that are present in gen_batch.non_tensor_batch but not in meta_batch to match concat in later operation
                            
                    response_batch = response_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    meta_batch = meta_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.meta_cog_n, interleave=True)
                    
                    response_batch_extra = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # Change the uid in batch to use in calculating meta score
                    batch.non_tensor_batch["uid"] = np.array(
                        [
                            uid + "_meta_cog" for uid in batch.non_tensor_batch["uid"]
                        ]
                    )
                    
                    meta_batch_extra = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.meta_cog_n, interleave=True)
                    if self.meta_feedin:
                        gen_batch = meta_batch
                    else:
                        batch = DataProto.concat([response_batch_extra, meta_batch_extra])
                        gen_batch = DataProto.concat([response_batch, meta_batch]) # raw_prompt length 32 is not equal to batch size 64
                else:
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    gen_batch = response_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            
                        if self.meta_feedin: # Gather meta cognition dicts from rollout output, store into each uid that corresponds to each sample in response_batch
                            import random
                            from copy import deepcopy
                            from verl.utils.reward_score.utils import extract_last_json_object
                            from .reward_score.meta_cog import is_json_text
                            from verl.utils.dataset.rl_dataset import collate_fn
                            from verl.utils.model import compute_position_id_with_mask
                            
                            
                            meta_dict_per_uid = defaultdict(list)
                            for idx, uid in enumerate(meta_batch_extra.non_tensor_batch["uid"]):
                                
                                prompt_id = gen_batch_output.batch["prompts"][idx]
                                response_id = gen_batch_output.batch["responses"][idx]
                                attention_mask = gen_batch_output.batch["attention_mask"][idx]
                                
                                prompt_length = prompt_id.shape[-1]
                                valid_response_length = attention_mask[prompt_length:].sum(
                                )
                                valid_response_ids = response_id[:valid_response_length]
                                meta_response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                                meta_json_dict = extract_last_json_object(meta_response_str)
                                meta_dict_per_uid[uid].append(meta_json_dict)

                            row_dict_list = []
                            for idx, uid in enumerate(response_batch_extra.non_tensor_batch["uid"]):
                                meta_dict_list = meta_dict_per_uid[uid+"_meta_cog"]
                                
                                if not len(meta_dict_list):
                                    continue
                                row_dict = response_batch[idx].non_tensor_batch.copy() # repeat operation do shallow copy
                                raw_prompt = row_dict["raw_prompt"]
                                selected_meta = random.choice(meta_dict_list) # fix to use all meta_dict_list exhaustively, and recycle else
                                
                                meta_sentence = ""
                                if len(meta_dict_list) == 0:
                                    pass
                                else:
                                    if "math_notion" in selected_meta:
                                        notions = selected_meta["math_notion"]
                                        meta_sentence += f" The problem could be solved using following math notions: {notions}. "
                                    if "problem_difficulty" in selected_meta:
                                        difficulty = selected_meta["problem_difficulty"]
                                        meta_sentence += f" The problem difficulty is {difficulty} out of 10."
                                    if "solution_length" in selected_meta:
                                        solution_length = selected_meta["solution_length"]
                                        meta_sentence += f" The solution length in token level should be {solution_length}."
                                
                                prompt = raw_prompt[0]["content"] + "\n\n" + meta_sentence
                                row_dict["raw_prompt"] = [{"role": "user", "content": prompt}]
                                raw_prompt = self.tokenizer.apply_chat_template(
                                    row_dict["raw_prompt"], add_generation_prompt=True, tokenize=False)

                                model_inputs = self.tokenizer(
                                    raw_prompt, return_tensors="pt", add_special_tokens=False)
                                input_ids = model_inputs.pop("input_ids")
                                attention_mask = model_inputs.pop("attention_mask")
                                input_ids, attention_mask = verl_F.postprocess_data(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_length=self.config.data.max_prompt_length,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    left_pad=True,
                                    truncation=self.config.data.truncation,
                                )
                                
                                raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
                                position_ids = compute_position_id_with_mask(attention_mask)

                                row_dict["input_ids"] = input_ids[0]
                                row_dict["attention_mask"] = attention_mask[0]
                                row_dict["position_ids"] = position_ids[0]
                                row_dict["raw_prompt_ids"] = raw_prompt_ids
                                row_dict_list.append(row_dict)
                            
                            single_dict = collate_fn(row_dict_list)
                            response_batch_meta_fedin = DataProto.from_single_dict(single_dict)
                            response_batch_output = self.actor_rollout_wg.generate_sequences(response_batch_meta_fedin)
                            response_batch_extra.meta_info = {}
                            
                            meta_batch_extra.union(gen_batch_output)
                            response_batch_extra.union(response_batch_output)
                            
                            batch = DataProto.concat([response_batch_extra, meta_batch_extra]) # check if uid is still intact
                        else:
                            batch.union(gen_batch_output)
                    
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(
                                gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(
                                dim=-1)

                            batch.pop(batch_keys=list(
                                gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(
                                batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict, all_uid_meta_info = compute_reward(
                                batch, self.reward_fn)
                    
                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(
                                rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(
                                rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(
                                rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(
                                rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(
                                rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                    batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(
                                    batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict, all_uid_meta_info = ray.get(
                                future_reward)

                        
                        batch.batch["token_level_scores"] = reward_tensor
                        # uid_track_reward
                        if reward_extra_infos_dict:
                            if self.is_meta_cog:
                                metrics.update({"critic/"+k+"_mean": np.array(v).mean() for k, v in reward_extra_infos_dict.items() if k.startswith("meta_cog")})
                                metrics.update({"critic/"+k+"_min": np.array(v).min() for k, v in reward_extra_infos_dict.items() if k.startswith("meta_cog")})
                                metrics.update({"critic/"+k+"_max": np.array(v).max() for k, v in reward_extra_infos_dict.items() if k.startswith("meta_cog")})
                            else:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        ) # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(
                                batch)
                            
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    
                    if all_uid_meta_info and self.config.actor_rollout_ref.get("meta_sft"):
                        sorted_meta_info = {
                            uid: sorted(items, key=lambda x: x["notion_score"], reverse=True)
                            for uid, items in all_uid_meta_info.items()
                        }
                        
                        for uid, items in sorted_meta_info.items():
                            if items[0]["notion_score"] < 0.2:
                                continue
                            prompts.append(items[0]["prompt"])
                            best_response = "<meta>" + items[0]["meta_reasoning"] + "</meta>" + "\n" + str(items[0]["meta_json"])
                            responses.append(best_response)
                        if len(prompts) >= self.config.actor_rollout_ref.meta_sft.data.train_batch_size:
                            print("[Expert SFT Best Response]: ", responses[0])
                            self.actor_rollout_wg.meta_sft_fit(
                                prompts[:self.config.actor_rollout_ref.meta_sft.data.train_batch_size],
                                responses[:self.config.actor_rollout_ref.meta_sft.data.train_batch_size],
                                self.config.actor_rollout_ref.meta_sft,
                                logger=logger,
                            )
                            prompts = prompts[self.config.actor_rollout_ref.meta_sft.data.train_batch_size:]
                            responses = responses[self.config.actor_rollout_ref.meta_sft.data.train_batch_size:]
                            
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get(
                        "rollout_data_dir", None)
                    
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print(
                                "Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(
                    self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(
                    batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(
                    batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(
                    batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
