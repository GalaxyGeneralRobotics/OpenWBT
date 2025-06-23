import time
import numpy as np
import onnxruntime as ort
import torch

from deploy.helpers.gait_planner import BipedalGaitPlanner
from deploy.config import Config
from typing import TypedDict

from deploy.helpers.rotation_helper import broadcast_quat_apply_np, broadcast_quat_multiply_np, quat_inv_np


class TaskDict(TypedDict):
    left_right_command: np.ndarray
    desired_pose: np.ndarray
    target_point: np.ndarray
    pose_reach_buf: np.ndarray
    target_point: np.ndarray
    gait_indices: np.ndarray
    clock_inputs: np.ndarray


class OdomInfo(TypedDict):
    root_pos: np.ndarray
    root_quat: np.ndarray


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_policy(policy_path):
    # load policy
    if policy_path.endswith(".pt"):  # jit
        policy_type = "jit"
        policy = torch.jit.load(policy_path)
    elif policy_path.endswith(".onnx"):  # onnx
        policy_type = "onnx"
        policy = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    else:
        raise NotImplementedError
    return policy_type, policy


class SquatLowLevelPolicy:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.policy_type, self.policy_session = load_policy(cfg.policy_path)

        # buffer
        self._last_action = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self.hidden_states = np.zeros([1, 1, 256], dtype=np.float32)

    def inference(self, cmd, gravity_orientation, omega, qj, dqj, upper_action=None):
        obs = self.compute_observation(cmd, gravity_orientation, omega, qj, dqj)
        if self.policy_type == "onnx":
            actions, self.hidden_states = self.policy_session.run(
                ["action", "output_hidden_states"],
                {
                    "obs": obs,
                    "input_hidden_states": self.hidden_states,
                },
            )
            action = actions[0].squeeze()
        else:
            raise NotImplementedError
        clip_actions = self.cfg.clip_actions
        action = np.clip(action, -clip_actions, clip_actions)
        self._last_action[:] = action.copy()
        assert action.shape[0] == self.cfg.num_actions
        # print(high_cmd[2:])
        if upper_action is not None:
            action = np.concatenate([action, upper_action])
        # print(high_cmd[2:])
        return obs, action, self.action_to_target_dof_pos(action)

    def compute_observation(self, cmd, gravity_orientation, omega, qj, dqj):
        default_angles_obs = self.cfg.default_angles[self.cfg.dof_idx]
        obs_cmd = cmd * self.cfg.cmd_scale
        obs_omega = omega * self.cfg.ang_vel_scale
        obs_qj = (qj - default_angles_obs) * self.cfg.dof_pos_scale
        obs_dqj = dqj * self.cfg.dof_vel_scale
        assert obs_qj.shape[0] == self.cfg.num_dof
        # update gait
        obs = np.concatenate([obs_cmd, gravity_orientation, obs_omega, obs_qj, obs_dqj, self._last_action])

        assert obs.shape[0] == self.cfg.num_obs
        clip_obs = self.cfg.clip_observations
        obs_buf = np.clip(obs, -clip_obs, clip_obs)
        obs_buf = obs_buf.astype(np.float32).reshape(1, -1)
        return obs_buf

    def action_to_target_dof_pos(self, action):
        action_scale = self.cfg.action_scale
        default_angles_action = self.cfg.default_angles[self.cfg.action_idx]
        target_dof_pos = action * action_scale + default_angles_action
        return target_dof_pos


class LocoLowLevelPolicy:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.policy_type, self.policy_session = load_policy(cfg.policy_path)

        # buffer
        self._last_action = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self.hidden_states = np.zeros([1, 1, 256], dtype=np.float32)

        self.gait_parameters = cfg.gait_parameters
        self.gait_planner = BipedalGaitPlanner(
            cfg.control_dt,
            self.gait_parameters["frequencies"],
            stance_ratio=self.gait_parameters["stance_ratio"],
            phase_offset=self.gait_parameters["phase_offset"],
        )

    def inference(self, cmd, gravity_orientation, omega, qj, dqj, upper_action=None):
        # breakpoint()
        obs = self.compute_observation(cmd, gravity_orientation, omega, qj, dqj)
        if self.policy_type == "onnx":
            actions, self.hidden_states = self.policy_session.run(
                ["action", "output_hidden_states"],
                {
                    "obs": obs,
                    "input_hidden_states": self.hidden_states,
                },
            )
            action = actions[0].squeeze()
        else:
            raise NotImplementedError
        # breakpoint()
        clip_actions = self.cfg.clip_actions
        action = np.clip(action, -clip_actions, clip_actions)
        self._last_action[:] = action.copy()
        assert action.shape[0] == self.cfg.num_actions
        # print(high_cmd[2:])
        if upper_action is not None:
            action = np.concatenate([action, upper_action])
        # print(high_cmd[2:])
        return obs, action, self.action_to_target_dof_pos(action)

    def compute_observation(self, cmd, gravity_orientation, omega, qj, dqj):
        default_angles_obs = self.cfg.default_angles[self.cfg.dof_idx]
        obs_cmd = cmd * self.cfg.cmd_scale
        obs_omega = omega * self.cfg.ang_vel_scale
        obs_qj = (qj - default_angles_obs) * self.cfg.dof_pos_scale
        obs_dqj = dqj * self.cfg.dof_vel_scale
        obs_clock = self.gait_planner.clock_inputs
        # print(obs_clock)
        obs = np.concatenate([obs_cmd, gravity_orientation, obs_omega, obs_qj, obs_dqj, self._last_action, obs_clock])

        assert obs_qj.shape[0] == self.cfg.num_dof
        assert obs.shape[0] == self.cfg.num_obs
        clip_obs = self.cfg.clip_observations
        obs_buf = np.clip(obs, -clip_obs, clip_obs)
        obs_buf = obs_buf.astype(np.float32).reshape(1, -1)
        return obs_buf

    def action_to_target_dof_pos(self, action):
        action_scale = self.cfg.action_scale
        default_angles_action = self.cfg.default_angles[self.cfg.action_idx]
        target_dof_pos = action * action_scale + default_angles_action
        return target_dof_pos
