from __future__ import division, print_function, absolute_import
from builtins import range
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json
# from ament_index_python.packages import get_package_share_directory


class ImitationDataset(Dataset):
    def __init__(self, device='cuda', is_val=False, is_test=False, mode='lstm'):
        super(ImitationDataset, self).__init__()

        # Initializing attributes
        self.poses = None
        self.scans = None
        self.velocities = None
        self.device = device

        # Get package path using ROS 2 equivalent function
        # pkg_path = get_package_share_directory('imitation_learning')
        file_path = '/home/hisham246/uwaterloo/robohub/imitation_learning_tb4/data/diffusion'
        print(f"Looking for files in: {file_path}")

        # Iterate through all data files
        i = 0
        while True:
            data_file = os.path.join(file_path, f'trajectory_{i}.json')
            if not os.path.exists(data_file):
                # Stop if no more files are found
                break

            # Open each data file
            with open(data_file, 'r') as file:
                traj_dict = json.load(file)

            # Convert the relevant parts to corresponding numpy arrays
            traj_pos = np.array(list(traj_dict.items())[0][-1])
            traj_scan = np.array(list(traj_dict.items())[1][-1])
            traj_vel = np.array(list(traj_dict.items())[2][-1])

            # Concatenate new data
            if i == 0:
                self.poses, self.scans, self.velocities = traj_pos, traj_scan, traj_vel
            else:
                self.poses = np.concatenate((self.poses, traj_pos), axis=0)
                self.scans = np.concatenate((self.scans, traj_scan), axis=0)
                self.velocities = np.concatenate((self.velocities, traj_vel), axis=0)
            i += 1

        # Check if any data was loaded
        if self.poses is None or self.scans is None or self.velocities is None:
            raise ValueError("No trajectory files found in the specified directory.")

        try:
            # Preprocess scans
            self.scans[self.scans == np.inf] = 3.5
            print(self.scans.shape)
            self.scans = self.scans[:, ::30]
        except Exception as e:
            print('Error processing scans:', e)

        print('Dataset loaded:', self.poses.shape[0])
        print("Laser scan observation shape:", self.scans.shape)

        if not is_test:
            # Separate into train or test set (80%-20%)
            if is_val:
                start, end = -int(self.poses.shape[0] * 0.2), -1
            else:
                start, end = 0, -int(self.poses.shape[0] * 0.2)

            # Convert numpy to tensors
            self.poses = torch.tensor(self.poses[start:end]).to(self.device, dtype=torch.float32)
            self.scans = torch.tensor(self.scans[start:end]).to(self.device, dtype=torch.float32)
            self.velocities = torch.tensor(self.velocities[start:end]).to(self.device, dtype=torch.float32)

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, i):
        pose = self.poses[i, :]
        scan = self.scans[i, :]
        vel = self.velocities[i, :]
        
        # Print the shape of laser_scan (scan) observation
        # print(f"Laser scan shape at index {i}: {scan.shape}")
        
        return pose, scan, vel


# Unit test
def unitTest():
    try:
        train_data = ImitationDataset(mode='lstm')
        test_data = ImitationDataset(is_val=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True)
        print("Dataset loaded successfully.")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    unitTest()
