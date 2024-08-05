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

		for path in all_files:
			# all entries of this file
			data = read_file(path, delim)
			print(data)


	def __len__(self):
		pass

	def __getitem__(self, idx):
		pass