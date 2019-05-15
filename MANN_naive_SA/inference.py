#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : inference.py.py
# @Author: harry
# @Date  : 2019/4/23 下午12:13
# @Desc  : MANN inference with eager execution enabled

import tensorflow as tf
import numpy as np
from mann import MANNModel
from constants import *
from hyper_params import *
from utils import *

tf.enable_eager_execution()


def main():
    print("Enable Eager Execution: {}".format(tf.executing_eagerly()))

    # build model
    model = MANNModel(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        training=False,
    )

    # load saved weights
    # model.load_weights(SAVE_FILE)
    model.load_weights(SAVE_BEST_FILE).expect_partial()

    # load word2id
    word2id = load_word2id()

    while True:
        try:
            # input data
            e1_text = input("Input exercise 1: ")
            e2_text = input("Input exercise 2: ")

            # tokenize
            e1_tokens = tokenize_raw_text_to_id(word2id, e1_text)
            e1_tokens = pad_tokens(word2id, e1_tokens)
            e2_tokens = tokenize_raw_text_to_id(word2id, e2_text)
            e2_tokens = pad_tokens(word2id, e2_tokens)

            print(e1_tokens)
            print(e2_tokens)

            # data for inference
            a = np.array([e1_tokens], dtype=int)
            b = np.array([e2_tokens], dtype=int)

            sim_score, sim_attention_matrix = model.predict([a, b])
            # print(repr(sim_score))
            print("Similar score: {}".format(sim_score[0][0]))
            print(repr(sim_attention_matrix))
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    main()
