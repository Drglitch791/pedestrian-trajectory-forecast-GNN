import os
import math
import numpy as np
from itertools import cycle

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data

import pandas as pd

def read_file(_path, delim='\t'):
	data = []
	with open(_path, 'r') as f:
		for line in f:
			line = line.strip().split(delim)
			line = [float(i) for i in line]
			data.append(line)
	return np.asarray(data)

class TrajectoryDataset(Dataset):
	def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_peds=1, max_peds=20, delim='\t'):
		super(TrajectoryDataset, self).__init__()

		'''contains all sequence
		sequence_list = [
			seq1: [
				ped1: [
					[...x...], [...y...] # each with length seq_len
				],
				ped2: [
					[...x...], [...y...]
				]
			],
			seq2: [
				ped1: [
					[...x...], [...y...]
				],
				ped2: [
					[...x...], [...y...]
				],
				....
			],
			....
		]
		'''
		sequence_list = []
		list_of_num_peds_in_sequences = []

		self.data_dir = data_dir
		self.obs_len = obs_len
		self.pred_len = pred_len
		self.seq_len = self.obs_len + self.pred_len

		all_files = sorted(os.listdir(self.data_dir))
		all_files = [os.path.join(self.data_dir, path) for path in all_files]

		self.max_peds_in_frame = 0

		for path in all_files:
			data = read_file(path, delim)
			frames = np.unique(data[:, 0]).tolist()
			frame_data = []
			for frame in frames:
				frame_data.append(data[frame == data[:, 0], :])
			num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
			
			# loop for every sequence
			for idx in range(0, num_sequences*skip+1, skip):
				curr_seq_data = np.concatenate(frame_data[idx:idx+self.seq_len], axis=0)
				peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
				self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))

				curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))

				# for every pedestrian in that sequence
				num_peds_considered = 0
				for _, ped_id in enumerate(peds_in_curr_seq):
					curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
					curr_ped_seq = np.around(curr_ped_seq, decimals=4)
					pad_front = frames.index(curr_ped_seq[0, 0]) - idx
					pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

					if (len(curr_ped_seq) != self.seq_len):
						continue

					curr_ped_seq = np.transpose(curr_ped_seq[:, 3:])
					_idx = num_peds_considered

					curr_seq[_idx] = curr_ped_seq
					num_peds_considered += 1
				
				if num_peds_considered > min_peds:
					list_of_num_peds_in_sequences.append(num_peds_considered)
					sequence_list.append(curr_seq[:num_peds_considered])

		self.num_seq = len(sequence_list)
		
		for seqidx, seq in enumerate(sequence_list):
			for frmidx, frm in enumerate(range(self.obs_len)):
				features_x = torch.zeros((len(seq), 2))
				for pedidx, ped in enumerate(seq):
					print(frmidx, seq)
					features_x[pedidx, : ] = torch.tensor([ped[0][frmidx], ped[1][frmidx]])
					edge_indices = []
					for e1 in range(len(seq)):
						for e2 in range(len(seq)):
							edge_indices.append([e1,e2])
				print(features_x)	
				break
			break
