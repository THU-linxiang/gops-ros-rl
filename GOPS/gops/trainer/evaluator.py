#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluation of trained policy
#  Update Date: 2021-05-10, Yang Guan: renew environment parameters


import numpy as np
import torch

from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_alg import create_approx_contrainer
from gops.utils.common_utils import set_seed


class Evaluator:
    def __init__(self, index=0, **kwargs):
        env_kwargs = kwargs.copy()
        env_kwargs.update({
            "reward_scale": None,
            "repeat_num": None,
            "gym2gymnasium": False,
            "vector_env_num": None,
        })
        evaluator_env_id = env_kwargs.get("evaluator_env_id")
        if evaluator_env_id is not None:
            env_kwargs["env_id"] = evaluator_env_id
        evaluator_bridge_env_id = env_kwargs.get("evaluator_bridge_env_id")
        if evaluator_bridge_env_id is not None:
            env_kwargs["bridge_env_id"] = evaluator_bridge_env_id
        else:
            env_kwargs.setdefault("bridge_env_id", "evaluator")

        self.env = create_env(**env_kwargs)

        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 400, self.env)

        self.networks = create_approx_contrainer(**kwargs)
        self.render = kwargs["is_render"]

        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
        self.eval_save = kwargs.get("eval_save", True)

        self.print_time = 0
        self.print_iteration = -1

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def run_an_episode(self, iteration, render=True):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        obs_list = []
        action_list = []
        reward_list = []
        reward_term_sums = {}
        reward_term_counts = {}
        obs, info = self.env.reset()
        done = 0
        info["TimeLimit.truncated"] = False
        while not (done or info["TimeLimit.truncated"]):
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            # 取众数，不随机探索
            action = action.detach().numpy()[0]
            next_obs, reward, done, next_info = self.env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            obs = next_obs
            info = next_info
            reward_terms = info.get("reward_terms", {})
            if isinstance(reward_terms, dict):
                for k, v in reward_terms.items():
                    if np.isscalar(v):
                        reward_term_sums[k] = reward_term_sums.get(k, 0.0) + float(v)
                        reward_term_counts[k] = reward_term_counts.get(k, 0) + 1
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                self.env.render()
            reward_list.append(reward)
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "obs_list": obs_list,
        }
        if self.eval_save:
            np.save(
                self.save_folder
                + "/evaluator/iter{}_ep{}".format(iteration, self.print_time),
                eval_dict,
            )
        episode_return = sum(reward_list)
        episode_reward_terms = {
            k: reward_term_sums[k] / max(1, reward_term_counts[k])
            for k in reward_term_sums.keys()
        }
        return episode_return, episode_reward_terms

    def run_n_episodes(self, n, iteration):
        episode_return_list = []
        reward_terms_list = []
        for _ in range(n):
            episode_return, episode_reward_terms = self.run_an_episode(iteration, self.render)
            episode_return_list.append(episode_return)
            reward_terms_list.append(episode_reward_terms)

        merged_reward_terms = {}
        all_term_names = set()
        for terms in reward_terms_list:
            all_term_names.update(terms.keys())
        for term_name in all_term_names:
            values = [terms[term_name] for terms in reward_terms_list if term_name in terms]
            if values:
                merged_reward_terms[term_name] = float(np.mean(values))

        return {
            "total_avg_return": float(np.mean(episode_return_list)),
            "reward_terms": merged_reward_terms,
        }

    def run_evaluation(self, iteration):
        return self.run_n_episodes(self.num_eval_episode, iteration)
