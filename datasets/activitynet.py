""" Dataset loader for the ActivityNet Captions dataset """
import os
import json

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from .BaseDataset import BaseDataset
from . import average_to_fixed_length
from core.config import config


class ActivityNet(BaseDataset):
    def __init__(self, split):
        # statistics for all video length
        # min:2 medium: max:1415  mean: 204, std:120
        # max sentence length：train-->73, test-->73
        super(ActivityNet, self).__init__(split)

    def __len__(self):
        return len(self.annotations)

    # def get_video_features(self, vid):
    #     with h5py.File(
    #             os.path.join(self.data_dir, 'sub_activitynet_v1-3.c3d.hdf5'),
    #             'r') as f:
    #         features = torch.from_numpy(f[vid]['c3d_features'][:])
    #     if config.DATASET.NORMALIZE:
    #         features = F.normalize(features, dim=1)
    #     vis_mask = torch.ones((features.shape[0], 1))
    #     return features, vis_mask

    def get_annotation(self):
        all_ids = os.listdir(self.feature_dirs['ActivityNet'])
        all_ids = [s[:-4] for s in all_ids]  # remove ".npy" suffix

        # val_1.json is renamed as val.json, val_2.json is renamed as test.json
        with open(
                os.path.join(self.anno_dirs['ActivityNet'],
                             '{}.json'.format(self.split)), 'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        # max_sentence_length = 0
        for vid, video_anno in annotations.items():
            if not vid in all_ids:
                # print(vid, '{}.json'.format(self.split))
                continue
            duration = video_anno['duration']
            for timestamp, sentence in zip(video_anno['timestamps'],
                                           video_anno['sentences']):
                # words = sentence.split()
                # if len(words) > max_sentence_length:
                #     max_sentence_length = len(words)
                # if timestamp[0] < timestamp[1] and len(words) <= 20:
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append({
                        'video':
                        vid,
                        'duration':
                        duration,
                        'times':
                        [max(timestamp[0], 0),
                         min(timestamp[1], duration)],
                        'description':
                        sentence,
                        'dataset':
                        'ActivityNet'
                    })
        # print("activitynet max sentence length: ", max_sentence_length)
        return anno_pairs