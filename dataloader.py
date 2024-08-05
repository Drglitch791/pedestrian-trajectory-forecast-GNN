import os
import math
import numpy as np
from itertools import cycle

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data


def read_file(_path, delim='\t'):
	data = []
	with open(_path, 'r') as f:
		for line in f:
			line = line.strip().split(delim)
			line = [float(i) for i in line]
			data.append(line)
	return np.asarray(data)

class TrajectoryDataset(Dataset):
	def __init__(self, data_dir, obs_len=8, pred_len=12, delim='\t'):
		super(TrajectoryDataset, self).__init__()

		self.data_dir = data_dir
		self.obs_len = obs_len
		self.pred_len = pred_len
		self.seq_len = self.obs_len + self.pred_len

		all_files = sorted(os.listdir(self.data_dir))
		all_files = [os.path.join(self.data_dir, path) for path in all_files]

		self.X = []
		self.Y = []

		for path in all_files:
			# all entries of this file
			data = read_file(path, delim)

			frames_list = np.sort(np.unique(data[:, 0]))
			peds_list = np.sort(np.unique(data[:, 1]))

			# collect history of each pedestrian
			peds_history = {}
			for ped in peds_list:
				peds_history[ped] = data[data[:, 1] == ped]

			# for every frame get past info of all pedestrians in that frame
			for fidx, frame in enumerate(frames_list):
				frame_data = data[data[:, 0] == frame]
				peds_in_frame = np.sort(np.unique(frame_data[:, 1]))

				frame_tensor_x = np.zeros((peds_in_frame.size, self.obs_len, 5))
				frame_tensor_y = np.zeros((peds_in_frame.size, self.pred_len, 2))
				for pidx, ped in enumerate(peds_in_frame):
					ped_history = peds_history[ped]
					frames_mask_x = frames_list[fidx-self.obs_len+1:fidx+1]
					frames_mask_y = frames_list[fidx+1:fidx+self.pred_len+1]
					
					# ped sequence wrt this frame
					ped_sequence_x = ped_history[np.isin(ped_history[:, 0], frames_mask_x)]
					ped_sequence_y = ped_history[np.isin(ped_history[:, 0], frames_mask_y)][:,-2:]
					if len(ped_sequence_x) + len(ped_sequence_y) == self.seq_len:
						frame_tensor_x[pidx] = ped_sequence_x
						frame_tensor_y[pidx] = ped_sequence_y
						self.X.append(frame_tensor_x)
						self.Y.append(frame_tensor_y)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		edge_index = []
		lenX = len(self.X[idx])
		for i in range(lenX):
			for j in range(lenX):
				edge_index.append([i,j])
		data = Data(
			x=torch.permute(torch.tensor(self.X[idx]), (0,2,1)), 
			y=torch.permute(torch.tensor(self.Y[idx]), (0,2,1)), 
			edge_index=torch.tensor(edge_index).T)
		return data