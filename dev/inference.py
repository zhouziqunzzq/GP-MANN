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
    model.load_weights(SAVE_FILE)

    # data for inference
    a = np.array([[1711, 4008, 2]], dtype=int)
    b = np.array([[1711, 4008, 2]], dtype=int)

    sim_score = model.predict([a, b])
    print(repr(sim_score))

    return


if __name__ == '__main__':
    main()
