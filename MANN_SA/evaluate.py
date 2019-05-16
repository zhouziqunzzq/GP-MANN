#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate.py
# @Author: harry
# @Date  : 2019/5/5 下午8:07
# @Desc  : Evaluate MANN

import tensorflow as tf
import numpy as np
from mann import MANNModel
from data_utils import load_evaluation_dataset
from constants import *
from hyper_params import *

tf.enable_eager_execution()


def evaluate():
    print("Enable Eager Execution: {}".format(tf.executing_eagerly()))

    dataset_sim, dataset_dis = load_evaluation_dataset(TRAINING_DATA_FILE_LIST)

    model = MANNModel(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        training=False,
    )

    # load saved weights
    model.load_weights(SAVE_BEST_FILE).expect_partial()

    # evaluation
    pass_cnt = 0
    total = 0
    for data_sim, data_dis in zip(dataset_sim, dataset_dis):
        outputs_sim, _ = model.predict(data_sim)
        outputs_dis, _ = model.predict(data_dis)

        diff = outputs_sim - outputs_dis
        pass_cnt += np.sum(diff >= MARGIN)

        total += len(outputs_sim)

        print(".", end='', flush=True)

    print()
    print("acc: {} ({} / {})".format(pass_cnt / total, pass_cnt, total))

    return


if __name__ == "__main__":
    evaluate()
