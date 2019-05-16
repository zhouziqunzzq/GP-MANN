#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : constants.py
# @Author: harry
# @Date  : 2019/4/23 下午12:14
# @Desc  : constant definition

VOCAB_PATH = './tf_data/word_list.txt'
VOCAB_SIZE = 4133
UNK_ID = 1
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
SEQ_LENGTH = 16

TRAINING_DATA_FILE_LIST = [
    # './tf_data/leetcode_pairwise_256.tfrecord',
    './tf_data/leetcode_pairwise_self_sim_256.tfrecord',
]

CPU_CORES = 4
PREFETCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 5120
BATCH_SIZE = 64

SAVE_PATH = './weights/'
SAVE_FILE = SAVE_PATH + 'MANN'
SAVE_BEST_PATH = './weights_best/'
SAVE_BEST_FILE = SAVE_BEST_PATH + 'MANN'
