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
from utils import convert_sparse

tf.enable_eager_execution()


# define parse functions
def merge_function(*args):
    return args


def get_parse_function(feature_name: str):
    assert feature_name in ['Text', 'Tokens', 'Tags', 'Similar Question Text',
                            'Similar Question Tokens', 'Similar Question Tags',
                            'Dissimilar Question Text', 'Dissimilar Question Tokens',
                            'Dissimilar Question Tags']

    def _parse_function(example_proto):
        features = {
            'Text': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'Tokens': tf.VarLenFeature(dtype=tf.int64),
            'Tags': tf.VarLenFeature(dtype=tf.int64),
            'Similar Question Text': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'Similar Question Tokens': tf.VarLenFeature(dtype=tf.int64),
            'Similar Question Tags': tf.VarLenFeature(dtype=tf.int64),
            'Dissimilar Question Text': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'Dissimilar Question Tokens': tf.VarLenFeature(dtype=tf.int64),
            'Dissimilar Question Tags': tf.VarLenFeature(dtype=tf.int64),
        }
        parsed_features = tf.parse_single_example(serialized=example_proto, features=features)
        convert_sparse(parsed_features, [
            'Tokens',
            'Tags',
            'Similar Question Tokens',
            'Similar Question Tags',
            'Dissimilar Question Tokens',
            'Dissimilar Question Tags',
        ])
        return parsed_features[feature_name]

    return _parse_function


# custom loss function
def triplet_loss(y_true, y_pred):
    # calculate triplet loss
    anchor_pos, anchor_neg = tf.split(y_pred, 2, axis=-1)
    loss = tf.math.maximum(0.0, MARGIN - anchor_pos + anchor_neg)
    loss = tf.math.reduce_sum(loss, axis=0)
    return loss


def main():
    print("Enable Eager Execution: {}".format(tf.executing_eagerly()))

    # prepare data
    dataset = tf.data.TFRecordDataset(filenames=TRAINING_DATA_FILE_LIST) \
        .prefetch(PREFETCH_SIZE)

    dataset_text = dataset.map(get_parse_function('Text'), num_parallel_calls=CPU_CORES)
    dataset_sim_text = dataset.map(get_parse_function('Similar Question Text'), num_parallel_calls=CPU_CORES)
    dataset_dis_text = dataset.map(get_parse_function('Dissimilar Question Text'), num_parallel_calls=CPU_CORES)

    dataset_text = dataset_text.batch(BATCH_SIZE)
    dataset_sim_text = dataset_sim_text.batch(BATCH_SIZE)
    dataset_dis_text = dataset_dis_text.batch(BATCH_SIZE)

    dataset_zipped = tf.data.Dataset.zip((dataset_text, dataset_sim_text, dataset_dis_text))
    dataset_merged = dataset_zipped.map(merge_function, num_parallel_calls=CPU_CORES)

    dummy_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([0], dtype=tf.int64)) \
        .repeat()

    dataset = tf.data.Dataset.zip(datasets=(dataset_merged, dummy_dataset)) \
        .shuffle(buffer_size=tf.constant(SHUFFLE_BUFFER_SIZE, dtype=tf.int64)) \
        .repeat() \
        .prefetch(PREFETCH_SIZE)

    # take a glance at what the dataset looks like
    # sum = 0
    # cnt = 0
    # my_max = 0
    # for x in dataset:
    #     # print(repr(x))
    #     inputs, _ = x
    #     a, b, c = inputs
    #     print(a.shape[1])
    #     print(b.shape[1])
    #     print(c.shape[1])
    #     target = c.shape[1]
    #     sum += int(target)
    #     my_max = max(my_max, target)
    #     cnt += 1
    # print(sum / cnt)
    # print(my_max)
    # return
    for x in dataset.take(1):
        print(type(x))
        print(repr(x))
    return


if __name__ == "__main__":
    main()
