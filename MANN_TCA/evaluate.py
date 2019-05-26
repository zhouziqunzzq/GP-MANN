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
from utils import *

tf.enable_eager_execution()


def evaluate():
    print("Enable Eager Execution: {}".format(tf.executing_eagerly()))

    dataset_sim, dataset_dis = load_evaluation_dataset(TRAINING_DATA_FILE_LIST)

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

    # plot pie
    n1 = pass_cnt
    n2 = total - pass_cnt
    p = plot_pie(
        labels=[u'正确 ({})'.format(n1), u'错误 ({})'.format(n2)],
        sizes=[n1, n2],
        colors=['grey', 'black'],
        explode=[0.1, 0]
    )
    p.show()

    return


if __name__ == "__main__":
    evaluate()
