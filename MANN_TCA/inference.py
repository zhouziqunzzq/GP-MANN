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
        embedding_size_vocab=EMBEDDING_SIZE_VOCAB,
        tag_size=TAG_SIZE,
        embedding_size_tag=EMBEDDING_SIZE_TAG,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        training=False,
    )

    # load saved weights
    # model.load_weights(SAVE_FILE)
    model.load_weights(SAVE_BEST_FILE).expect_partial()

    # load word2id
    word2id, id2word = load_word2id()

    # load tag2id
    tag2id = load_tag2id()

    # input data
    e1_text = input("Input exercise 1: ")
    e1_raw_tags = input("Input tags for exercise 1: ")
    e2_text = input("Input exercise 2: ")
    e2_raw_tags = input("Input tags for exercise 2: ")

    # tokenize
    e1_tokens = tokenize_raw_text_to_id(word2id, e1_text)
    e1_tags = tokenize_raw_tags_to_id(tag2id, e1_raw_tags)
    e2_tokens = tokenize_raw_text_to_id(word2id, e2_text)
    e2_tags = tokenize_raw_tags_to_id(tag2id, e2_raw_tags)

    # pad tags
    e1_tags, e2_tags = pad_tags(tag2id, e1_tags, e2_tags)

    print(e1_tokens)
    print(e1_tags)
    print(e2_tokens)
    print(e2_tags)

    # data for inference
    a_tokens = np.array([e1_tokens], dtype=int)
    a_tags = np.array([e1_tags], dtype=int)
    b_tokens = np.array([e2_tokens], dtype=int)
    b_tags = np.array([e2_tags], dtype=int)

    sim_score, sim_attention_matrix = model.predict([a_tokens, a_tags, b_tokens, b_tags])
    # print(repr(sim_score))
    print("Similar score: {}".format(sim_score[0][0]))
    valid_attention = sim_attention_matrix[0]
    # print(repr(valid_attention))
    e1_raw = [id2word[w] for w in e1_tokens]
    e2_raw = [id2word[w] for w in e2_tokens]
    plot_attention(valid_attention, e1_raw, e2_raw)


if __name__ == '__main__':
    main()
