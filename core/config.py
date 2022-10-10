from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

config = edict()

config.WORKERS = 16
config.LOG_DIR = ''
config.MODEL_DIR = ''
config.VERBOSE = False
config.TAG = None

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = ''
config.MODEL.CHECKPOINT = ''  # The checkpoint for the best performance
config.MODEL.PARAMS = None

# DATASET related params
config.DATASET = edict()
config.DATASET.NAME = ''
config.DATASET.DATASETS = []
config.DATASET.NO_VAL = True
config.DATASET.NUM_SAMPLE_CLIPS = 128
config.DATASET.SPLIT = ''
config.DATASET.NORMALIZE = False
config.DATASET.EXTEND_INNRE = 0.0  # extend the inner action label
config.DATASET.EXTEND_TIME = False  # extend TIME length of the input for bias
config.DATASET.FLIP_TIME = False  # flip the input in time direction
# train
config.TRAIN = edict()
config.TRAIN.LR = 0.001
config.TRAIN.WEIGHT_DECAY = 0.0001
config.TRAIN.FACTOR = 0.8
config.TRAIN.PATIENCE = 5
config.TRAIN.GAMMA = 0.5
config.TRAIN.MILE_STONE = [10, 15]
config.TRAIN.MAX_EPOCH = 20
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.PER_NEGATIVE_PAIRS_INBATCH = 3
config.TRAIN.SHUFFLE = True
config.TRAIN.CONTINUE = False

config.LOSS = edict()
config.LOSS.NAME = 'bce_loss'
config.LOSS.MATCH = 1.0
config.LOSS.DISTANCE = 1.0
config.LOSS.KL = 1.0
config.LOSS.EARLY = 1.0
config.LOSS.LOCALIZATION = 1.0
config.LOSS.CLIP_NORM = 1.0
config.LOSS.DCOR = 1.0
config.LOSS.PARAMS = None

# test
config.TEST = edict()
config.TEST.RECALL = []
config.TEST.TIOU = []
config.TEST.NMS_THRESH = 0.4
config.TEST.INTERVAL = 1
config.TEST.EVAL_TRAIN = False
config.TEST.BATCH_SIZE = 1
config.TEST.TOP_K = 10
config.TEST.SHUFFLE_VIDEO_FRAME = False


def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k], v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
