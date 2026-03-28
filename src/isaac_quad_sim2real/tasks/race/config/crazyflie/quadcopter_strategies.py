# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

        # Initialize buffer 
        self._initial_dist_to_goal = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._prev_dist_to_goal = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._prev_drone_x_wrt_gate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure

        # Check gate passing
        current_drone_x_wrt_gate = self.env._pose_drone_wrt_gate[:, 0]
        # 1. Check if the drone crossed the Y-Z plane of the gate
        plane_crossed = (self._prev_drone_x_wrt_gate > 0.0) & (current_drone_x_wrt_gate <= 0.0)
        plane_crossed_wrong = (self._prev_drone_x_wrt_gate <= 0.0) & (current_drone_x_wrt_gate > 0.0)
        # 2. Check if the drone was actually inside the gate's physical bounds when it crossed
        d = self.env._gate_model_cfg_data.gate_side / 2
        local_y = self.env._pose_drone_wrt_gate[:, 1]
        local_z = self.env._pose_drone_wrt_gate[:, 2]
        inside_bound = (torch.abs(local_y) < d) & (torch.abs(local_z) < d)
        # 3. Check if both the conditions are satisfied
        gate_passed = (plane_crossed & inside_bound)
        wrong_way_cross = (plane_crossed_wrong & inside_bound)
        ids_gate_passed = torch.where(gate_passed)[0]
        # 4. Update buffer
        self._prev_drone_x_wrt_gate = current_drone_x_wrt_gate.clone()

        # Update waypoints for environments where the quadcopters successfully passed the gate
        self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
        self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
        self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

        #==================================================================================================#
        #==================================== Gate Passing Reward (Sparse) ================================#
        #==================================================================================================#
        gate_pass_reward = gate_passed.float()
        wrong_direction_reward = wrong_way_cross.float()

        #==================================================================================================#
        #====================================== Progress Reward ===========================================#
        #==================================================================================================#
        current_dist_to_goal = torch.linalg.norm(self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1)
        # Update distance to goal if gate passed
        self._prev_dist_to_goal[ids_gate_passed] = current_dist_to_goal[ids_gate_passed]
        self._initial_dist_to_goal[ids_gate_passed] = current_dist_to_goal[ids_gate_passed]

        # progress_reward = self._prev_dist_to_goal - current_dist_to_goal
        
        delta = self._prev_dist_to_goal - current_dist_to_goal
        progress_reward = torch.clamp(delta, min=0.0) # clamped progress reward, do not penalize going away
        # Update previous distance to goal
        self._prev_dist_to_goal = current_dist_to_goal.clone()

        #==================================================================================================#
        #=========================================== Crash Reward =========================================#
        #==================================================================================================#
        # Compute crashed environments if contact detected for 100 timesteps
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 10).int()
        self.env._crashed = self.env._crashed + crashed * mask
        crash_reward = (crashed * mask).float()

        #==================================================================================================#
        #========================================== Velocity Reward =======================================#
        #==================================================================================================#

        velocity = self.env._robot.data.root_lin_vel_w
        direction = self.env._desired_pos_w - self.env._robot.data.root_link_pos_w
        direction = direction / torch.norm(direction, dim=1, keepdim=True)

        velocity_reward = torch.sum(velocity * direction, dim=1)
        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "gate_pass_reward": gate_pass_reward * self.env.rew['gate_pass_reward_scale'],
                "progress_reward": progress_reward * self.env.rew['progress_reward_scale'],
                "crash_reward": crash_reward * self.env.rew['crash_reward_scale'],
                "velocity_reward": velocity_reward * self.env.rew['velocity_reward_scale'],
                "wrong_direction_reward": wrong_direction_reward * self.env.rew['wrong_direction_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations
        #### Basic drone states, modify for your needs)
        drone_pose_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w

        # Linear and angular velocities (referred to as body rates)
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b  # [roll_rate, pitch_rate, yaw_rate]

        # Current target gate information
        current_gate_idx = self.env._idx_wp
        current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]  # World position of current gate
        current_gate_yaw = self.env._waypoints[current_gate_idx, -1]    # Yaw orientation of current gate

        # # Relative position to current gate in gate frame
        # drone_pos_gate_frame = self.env._pose_drone_wrt_gate

        # Relative position to current gate in body frame
        gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w,
            current_gate_pos_w
        )

        # Relative yaw to current gate in world frame
        _, _, drone_yaw = euler_xyz_from_quat(drone_quat_w)
        relative_yaw = wrap_to_pi(current_gate_yaw - drone_yaw).unsqueeze(-1)

        # Previous actions
        prev_actions = self.env._previous_actions  # Shape: (num_envs, 4)

        # Number of gates passed
        # gates_passed = self.env._n_gates_passed.unsqueeze(1).float()

        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                drone_quat_w,       # orientation in the world frame (4 dims)
                drone_lin_vel_b,    # linear velocity in the body frame (3 dims)
                drone_ang_vel_b,    # angular velocity in the body frame (3 dims)
                gate_pos_b,         # relative position to current gate in the body frame (3 dims)
                relative_yaw,       # relative yaw (1 dim)
                prev_actions        # previous action (4 dims)
            ],
            # TODO ----- END -----
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        # randomize starting gate
        num_waypoints = self.env._waypoints.shape[0]
        waypoint_indices = torch.randint(0, num_waypoints, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        z0_wp = self.env._waypoints[waypoint_indices][:, 2]
        theta = self.env._waypoints[waypoint_indices][:, -1]

        # TODO randomize initial positions
        x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, -0.5)
        base_noise = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)
        y_local = torch.abs(x_local) * 0.5 * base_noise
        z_local = torch.abs(x_local) * 0.3 * base_noise

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        # Ensure the initial positions inside the task environment
        initial_x = torch.clamp(x0_wp - x_rot, -3.0, 3.5)
        initial_y = torch.clamp(y0_wp - y_rot, -7.0, 7.0)
        initial_z = torch.clamp(z0_wp + z_local, 0.05, 3.5)

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # TODO point drone towards the gate with a random angle
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            wrap_to_pi(initial_yaw + torch.empty(n_reset, device=self.device).uniform_(-0.7854, 0.7854)) # [-pi/4, pi/4]
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0

        # Reset buffer
        dist_to_goal = torch.linalg.norm(self.env._desired_pos_w[env_ids] - self.env._robot.data.root_link_pos_w[env_ids], dim=1)
        self._initial_dist_to_goal[env_ids] = dist_to_goal.clone()
        self._prev_dist_to_goal[env_ids] = dist_to_goal.clone()
        self._prev_drone_x_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0].clone()
