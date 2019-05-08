#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate.py
# @Author: harry
# @Date  : 2019/5/5 下午8:07
# @Desc  : Evaluate LCS

import tensorflow as tf
import numpy as np
from lcs import lcs
from data_utils import load_evaluation_dataset
from constants import *

tf.enable_eager_execution()


def tensor_to_list(t):
    return list(t.numpy())


def get_seq_length(dataset):
    a_tokens, a_tags, b_tokens, b_tags = dataset
    return int(a_tokens.shape[0]), int(b_tokens.shape[0])


def evaluate_step_pair(a_tokens, a_tags, b_tokens, b_tags):
    return lcs(tensor_to_list(a_tokens), tensor_to_list(b_tokens))


def evaluate_step(data_sim, data_dis):
    result_sim = evaluate_step_pair(*data_sim)
    result_dis = evaluate_step_pair(*data_dis)
    return result_sim, result_dis


def evaluate():
    print("Enable Eager Execution: {}".format(tf.executing_eagerly()))

    dataset_sim, dataset_dis = load_evaluation_dataset(TRAINING_DATA_FILE_LIST)

    # evaluation
    total = 0
    diff_list = []
    for data_sim, data_dis in zip(dataset_sim, dataset_dis):
        result_sim, result_dis = evaluate_step(data_sim, data_dis)

        anchor_len, sim_len = get_seq_length(data_sim)
        _, dis_len = get_seq_length(data_dis)

        sim_score = len(result_sim) / max(anchor_len, sim_len)
        dis_score = len(result_dis) / max(anchor_len, dis_len)
        diff = sim_score - dis_score

        diff_list.append(diff)
        total += 1

        print(".", end='', flush=True)

    print()
    diff_array = np.array(diff_list, dtype=float)

    pass_cnt = np.sum(diff_array >= 0.5)
    print("acc(MARGIN=0.5): {} ({} / {})".format(pass_cnt / total, pass_cnt, total))
    pass_cnt = np.sum(diff_array >= 0.4)
    print("acc(MARGIN=0.5): {} ({} / {})".format(pass_cnt / total, pass_cnt, total))
    pass_cnt = np.sum(diff_array >= 0.3)
    print("acc(MARGIN=0.5): {} ({} / {})".format(pass_cnt / total, pass_cnt, total))
    pass_cnt = np.sum(diff_array >= 0.2)
    print("acc(MARGIN=0.5): {} ({} / {})".format(pass_cnt / total, pass_cnt, total))
    pass_cnt = np.sum(diff_array >= 0.1)
    print("acc(MARGIN=0.5): {} ({} / {})".format(pass_cnt / total, pass_cnt, total))

    return


if __name__ == "__main__":
    evaluate()
