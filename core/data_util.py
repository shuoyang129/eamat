import os
import glob
import json
import pickle
import numpy as np
from tqdm import tqdm


def load_json(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode='w', encoding='utf-8') as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_lines(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        return [e.strip("\n") for e in f.readlines()]


def save_lines(data, filename):
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write("\n".join(data))


def load_pickle(filename):
    with open(filename, mode='rb') as handle:
        data = pickle.load(handle)
        return data


def save_pickle(data, filename):
    with open(filename, mode='wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap


# def time_to_index(start_time, end_time, num_units, duration):
#     s_times = np.arange(0, num_units).astype(
#         np.float32) / float(num_units) * duration
#     e_times = np.arange(1, num_units + 1).astype(
#         np.float32) / float(num_units) * duration
#     candidates = np.stack([
#         np.repeat(s_times[:, None], repeats=num_units, axis=1),
#         np.repeat(e_times[None, :], repeats=num_units, axis=0)
#     ],
#                           axis=2).reshape((-1, 2))
#     overlaps = compute_overlap(candidates.tolist(),
#                                [start_time, end_time]).reshape(
#                                    num_units, num_units)
#     start_index = np.argmax(overlaps) // num_units
#     end_index = np.argmax(overlaps) % num_units
#     return start_index, end_index, overlaps


def index_to_time(start_index, end_index, num_units, extend_pre, extend_suf,
                  duration):
    if start_index <= extend_pre:
        start_index = extend_pre
    if end_index <= extend_pre:
        end_index = extend_pre
    s_times = np.arange(0, num_units).astype(
        np.float32) * duration / float(num_units)
    e_times = np.arange(1, num_units + 1).astype(
        np.float32) * duration / float(num_units)
    start_time = s_times[start_index - extend_pre]
    end_time = e_times[end_index - extend_pre]
    return start_time, end_time
