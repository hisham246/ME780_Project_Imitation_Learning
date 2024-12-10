#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm
import collections

import sys
sys.path.append('/home/hisham246/uwaterloo/robohub/imitation_learning_tb4')

# Import the diffusion model
from policy.transformer_for_diffusion import TransformerForDiffusion
from policy.diffusion_transformer import DiffusionTransformerLowdimPolicy
from policy.normalizer import LinearNormalizer

class DiffusionPolicyExecutor(Node):
    def __init__(self):
        super().__init__('diffusion_policy_executor')

        # Define parameters
        self.obs_horizon = 1
        self.pred_horizon = 10
        self.action_horizon = 5
        self.obs_dim = 367
        self.action_dim = 2
        self.num_diffusion_iters = 100
        self.var = 2

        self.laser_data = None
        self.pose = None

        # Load model and scheduler
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.transformer_model = TransformerForDiffusion(
            input_dim=self.action_dim,
            output_dim=self.action_dim,
            horizon=self.pred_horizon,
            n_obs_steps=self.obs_horizon,
            cond_dim=self.obs_dim * self.obs_horizon,
            n_layer=4,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.1,
            p_drop_attn=0.1,
            causal_attn=False,
            time_as_cond=True,
            obs_as_cond=True
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon'
        )
        self.policy = DiffusionTransformerLowdimPolicy(
            model=self.transformer_model,
            noise_scheduler=noise_scheduler,
            horizon=self.pred_horizon,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_action_steps=self.action_horizon,
            n_obs_steps=self.obs_horizon,
            obs_as_cond=True,
            pred_action_steps_only=False
        )

        ckpt_path = '/home/hisham246/uwaterloo/robohub/imitation_learning_tb4/models/diffusion/diffusion_policy_real_13.pt'
        self.policy.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        self.policy.to(device)
        self.policy.eval()

        self.normalizer = LinearNormalizer()
        normalizer_params = torch.load('/home/hisham246/uwaterloo/robohub/imitation_learning_tb4/models/diffusion/normalizer_params_real_13.pt', weights_only=False)
        self.normalizer.params_dict.update(normalizer_params)
        self.policy.set_normalizer(self.normalizer)

        self.obs_deque = collections.deque(maxlen=self.obs_horizon)

        # Publishers and subscribers
        qos_profile = QoSProfile(depth=10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', qos_profile)
        self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_profile)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, qos_profile)

        self.get_logger().info("Diffusion policy executor initialized.")

    def laser_callback(self, scan):
        self.laser_data = scan

    def pose_callback(self, amcl):
        self.pose = amcl

    def execute_policy(self):
        self.get_logger().info("Starting policy execution...")

        rate = self.create_rate(20)  # 10 Hz

        # Wait for sensor data
        while self.pose is None or self.laser_data is None:
            self.get_logger().info("Waiting for pose and laser data...")
            rclpy.spin_once(self, timeout_sec=0.1)

        odom_input = [
            self.pose.pose.pose.position.x, self.pose.pose.pose.position.y, self.pose.pose.pose.position.z,
            self.pose.pose.pose.orientation.x, self.pose.pose.pose.orientation.y, self.pose.pose.pose.orientation.z,
            self.pose.pose.pose.orientation.w
        ]

        ranges = np.array([self.laser_data.ranges])
        ranges[np.isinf(ranges)] = 3.5
        ranges = ranges[:,::self.var]

        laser_scan = np.array([ranges])

        initial_obs = np.concatenate([odom_input, laser_scan.flatten()])
        for _ in range(self.obs_horizon):
            self.obs_deque.append(initial_obs)

        while rclpy.ok():
            rclpy.spin_once(self)

            # Normalize and prepare observation
            obs_dict = {'obs': np.stack(list(self.obs_deque))}
            nobs = self.normalizer.normalize(obs_dict)['obs']
            # nobs = torch.tensor(nobs).unsqueeze(0).to(self.device, dtype=torch.float32)
            nobs = nobs.clone().detach().unsqueeze(0).to(self.device, dtype=torch.float32)
            obs_cond = nobs.flatten(start_dim=1).unsqueeze(1)

            noisy_action = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action
            self.policy.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            with torch.no_grad():
                for k in self.policy.noise_scheduler.timesteps:
                    noise_pred = self.policy.model(sample=naction, timestep=k, cond=obs_cond)
                    naction = self.policy.noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=naction
                    ).prev_sample

            naction = naction.cpu().numpy().squeeze(0)
            action_pred = self.normalizer.unnormalize({'action': naction})['action']
            action = action_pred[:self.action_horizon, :]

            # Publish commands
            for act in action:
                cmd_vel = Twist()
                cmd_vel.linear.x = float(act[0])
                cmd_vel.angular.z = float(act[1])
                self.pub_cmd_vel.publish(cmd_vel)
                self.get_logger().info(f"Published velocity: linear={act[0]}, angular={act[1]}")

                rclpy.spin_once(self, timeout_sec=0.1)

                if self.pose and self.laser_data:
                    odom_input = [
                        self.pose.pose.pose.position.x, self.pose.pose.pose.position.y, self.pose.pose.pose.position.z,
                        self.pose.pose.pose.orientation.x, self.pose.pose.pose.orientation.y, self.pose.pose.pose.orientation.z,
                        self.pose.pose.pose.orientation.w
                    ]
                    ranges = np.array([self.laser_data.ranges])
                    ranges[np.isinf(ranges)] = 3.5
                    ranges = ranges[:,::self.var]
                    laser_scan = np.array([ranges])
                    current_obs = np.concatenate([odom_input, laser_scan.flatten()])
                    self.obs_deque.append(current_obs)

            rate.sleep()


def main(args=None):
    rclpy.init(args=args)

    executor = None

    try:
        executor = DiffusionPolicyExecutor()
        executor.execute_policy()
    except Exception as e:
        if executor:
            executor.get_logger().error(f"Execution error: {str(e)}")
        else:
            print(f"Execution error (during initialization): {str(e)}")
    finally:
        if executor:
            executor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
