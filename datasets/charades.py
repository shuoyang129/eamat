""" Dataset loader for the Charades-STA dataset """
import os
import csv

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from . import average_to_fixed_length
from .BaseDataset import BaseDataset
from core.config import config


class Charades(BaseDataset):
    def __init__(self, split):
        # statistics for all video length
        # min:12 max:390 mean: 62, std:18
        # max sentence length：train->10, test->10
        super(Charades, self).__init__(split)

    def __len__(self):
        return len(self.annotations)

    def get_annotation(self):
        self.durations = {}
        with open(
                os.path.join(self.anno_dirs['Charades'],
                             'Charades_v1_{}.csv'.format(self.split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])

        anno_file = open(
            os.path.join(self.anno_dirs['Charades'],
                         "charades_sta_{}.txt".format(self.split)), 'r')
        annotations = []
        # max_sentence_length = 0
        for line in anno_file:
            anno, sent = line.split("##")
            sent = sent.split('.\n')[0]
            vid, s_time, e_time = anno.split(" ")
            # words = sent.split()
            # if len(words) > max_sentence_length:
            #     max_sentence_length = len(words)
            s_time = float(s_time)
            e_time = min(float(e_time), self.durations[vid])
            if s_time < e_time:
                annotations.append({
                    'video': vid,
                    'times': [s_time, e_time],
                    'description': sent,
                    'duration': self.durations[vid],
                    'dataset': 'Charades'
                })
        anno_file.close()
        # print("charade max sentence length: ", max_sentence_length)
        return annotations