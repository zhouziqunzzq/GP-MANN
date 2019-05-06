#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: harry
# @Date  : 2019/4/12 下午2:59
# @Desc  : MANN training with eager execution enabled

import tensorflow as tf
import matplotlib.pyplot as plt
import os
from callbacks import MyModelCheckpoint, PrintLossCallback
from mann import MANNModel
from constants import *
from hyper_params import *
from data_utils import load_training_dataset

tf.enable_eager_execution()


def main():
    print("Enable Eager Execution: {}".format(tf.executing_eagerly()))

    # prepare data
    dataset = load_training_dataset(TRAINING_DATA_FILE_LIST)

    # take a glance at what the dataset looks like
    # for x in dataset.take(1):
    #     print(type(x))
    #     print(repr(x))
    # return

    # build model
    model = MANNModel(
        vocab_size=VOCAB_SIZE,
        embedding_size_vocab=EMBEDDING_SIZE_VOCAB,
        tag_size=TAG_SIZE,
        embedding_size_tag=EMBEDDING_SIZE_TAG,
        lstm_units=LSTM_UNITS,
        attention_units=ATTENTION_UNITS,
        dense_units=DENSE_UNITS,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)

    model.train(dataset=dataset, optimizer=optimizer, epochs=1)

    return


if __name__ == "__main__":
    main()
