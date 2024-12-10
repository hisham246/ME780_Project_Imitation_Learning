#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import json
import os
import argparse

class DataCollector(Node):
    def __init__(self, args):
        super().__init__('data_collection')
        
        # Parameters
        self.data_dir = args.data_dir
        self.device = args.device
        self.num_traj = 0

        # Data recording structure
        self.D = {
            'robot_pos': [],
            'laser_scan': [],
            'robot_vel': []
        }
        
        # ROS 2 subscribers
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.odom_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        self.laser_data = None
        self.odom_data = None
        self.cmd_vel_data = None
        
        # Timer for periodic data collection
        self.timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(self.timer_period, self.collect_data)
        
        # Initialize dataset directory
        os.makedirs(self.data_dir, exist_ok=True)
        self.get_logger().info("Data collection started...")

    def laser_callback(self, msg):
        """Callback for LaserScan data"""
        self.laser_data = np.array(msg.ranges).tolist()

    def odom_callback(self, msg):
        """Callback for Odometry data"""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.odom_data = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

    def cmd_vel_callback(self, msg):
        """Callback for Twist data (robot velocity)"""
        self.cmd_vel_data = [msg.linear.x, msg.angular.z]

    def collect_data(self):
        """Collect data from the latest messages and store in dataset."""
        if self.laser_data is not None and self.odom_data is not None and self.cmd_vel_data is not None:
            self.D["robot_pos"].append(self.odom_data)
            self.D["laser_scan"].append(self.laser_data)
            self.D["robot_vel"].append(self.cmd_vel_data)
            
            # Log every few messages
            if len(self.D["robot_pos"]) % 50 == 0:
                self.get_logger().info(f"Collected {len(self.D['robot_pos'])} samples")

    def save_data_to_file(self):
        """Save the collected data to a JSON file."""
        file_name = os.path.join(self.data_dir, f"trajectory_{self.num_traj}.json")
        while os.path.exists(file_name):
            self.num_traj += 1
            file_name = os.path.join(self.data_dir, f"trajectory_{self.num_traj}.json")
        
        with open(file_name, 'w') as f:
            json.dump(self.D, f)
            self.get_logger().info(f"Saved data to {file_name}")
        
        self.num_traj += 1
        self.clear_dataset()

    def clear_dataset(self):
        """Clear the current dataset."""
        self.D = {
            'robot_pos': [],
            'laser_scan': [],
            'robot_vel': []
        }
        self.get_logger().info("Cleared dataset for new collection")

def main(args=None):
    # Initialize ROS 2 Python client library
    rclpy.init(args=args)

    # Argument parsing
    parser = argparse.ArgumentParser(description="Data collection parameters")
    parser.add_argument('--data_dir', type=str, default=os.getcwd() + '/data/diffusion')
    parser.add_argument('--device', type=str, default='cpu')
    parsed_args = parser.parse_args()

    # Create the data collector node
    data_collector = DataCollector(parsed_args)

    # Run the node until manually terminated
    try:
        rclpy.spin(data_collector)
    except KeyboardInterrupt:
        data_collector.get_logger().info("Keyboard Interrupt - Shutting down")
    finally:
        # Save any remaining data before exiting
        if data_collector.D['robot_pos']:
            data_collector.save_data_to_file()
        data_collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
