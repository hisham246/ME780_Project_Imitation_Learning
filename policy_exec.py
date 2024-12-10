#!/usr/bin/env python3

import rclpy
import rclpy.logging
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

import numpy as np
import argparse
import os
from rospkg import RosPack

from model_il import ImitationNet
import model_il
import torch

from rclpy.qos import QoSProfile

qos_profile=QoSProfile(reliability=2, durability=2, history=1, depth=10)

class PolicyExecutor(Node):
    def __init__(self, model_dir, device):
        super().__init__('policy_executor')

        self.model_dir = model_dir
        self.device = device
        self.model = ImitationNet(control_dim=2, device=self.device)

        # Initialize variables
        self.laser = None
        self.pos = None

        # Publisher and subscribers
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_profile=qos_profile)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, qos_profile=qos_profile)

        # Load model
        if os.path.exists(self.model_dir + 'model_final.pt'):
            self.model.load(self.model_dir + 'model_final.pt')
        else:
            self.get_logger().error(f"Model file not found in {self.model_dir}")
            exit(1)

        self.model = self.model.to(self.device)
        self.model.eval()

    def laser_callback(self, scan):
        self.get_logger().info("Laser data received.")
        self.laser = scan

    def amcl_callback(self, amcl):
        if amcl is not None:
            self.get_logger().info(f"AMCL Pose Received: Position({amcl.pose.pose.position.x}, {amcl.pose.pose.position.y})")
            self.pos = amcl
        else:
            self.get_logger().warn("AMCL callback received empty data!")

    def execute(self):
        # Wait for initial data
        timeout = 60
        start_time = self.get_clock().now().seconds_nanoseconds()[0]
        while (self.laser is None or self.pos is None) and \
              (self.get_clock().now().seconds_nanoseconds()[0] - start_time < timeout):
            self.get_logger().info('Waiting for sensor data...')
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.laser is None or self.pos is None:
            self.get_logger().error("Timed out waiting for sensor data.")
            return

        self.get_logger().info('Starting policy execution.')

        rate = self.create_rate(20)  # 20 Hz
        robot_cmd = Twist()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.pos is None or self.laser is None:
                self.get_logger().warn('Lost sensor data. Waiting to recover...')
                continue

            try:
                # Prepare inputs for the model
                odom_input = np.array([[float(self.pos.pose.pose.position.x),
                                        float(self.pos.pose.pose.position.y),
                                        float(self.pos.pose.pose.position.z),
                                        float(self.pos.pose.pose.orientation.x),
                                        float(self.pos.pose.pose.orientation.y),
                                        float(self.pos.pose.pose.orientation.z),
                                        float(self.pos.pose.pose.orientation.w)]])
                laser_scan = np.array([self.laser.ranges])

                # Get velocity commands from the model
                vel = model_il.test(self.model, odom_input, laser_scan)
                self.get_logger().info(f"Model Output (vel): {vel}")

                # Publish velocity command
                robot_cmd.linear.x = float(vel[0, 0])
                robot_cmd.angular.z = float(vel[0, 1])
                self.pub_cmd.publish(robot_cmd)
            except Exception as e:
                self.get_logger().error(f"Error in policy execution: {str(e)}")
                continue


def main(args=None):
    rclpy.init(args=args)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--model_dir', type=str,
                        default='/home/hisham246/uwaterloo/robohub/imitation_learning_tb4/models/lstm/')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Configure CUDA if available
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = "cpu"

    # Create and run the policy executor node
    executor = PolicyExecutor(model_dir=args.model_dir, device=args.device)

    try:
        executor.execute()
    finally:
        executor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()