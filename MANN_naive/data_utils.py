#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_utils.py
# @Author: harry
# @Date  : 2019/5/2 下午11:19
# @Desc  : Utils for loading Dataset

import tensorflow as tf
from constants import *
from utils import convert_sparse


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


def get_max_length(*args) -> int:
    max_length = 0
    for dataset in args:
        for s in dataset:
            max_length = max(max_length, len(s))
    return max_length


def load_training_dataset(tfrecord_files: list) -> tf.data.Dataset:
    # prepare data
    dataset = tf.data.TFRecordDataset(filenames=tfrecord_files) \
        .prefetch(PREFETCH_SIZE)

    dataset_tokens = dataset.map(get_parse_function('Tokens'), num_parallel_calls=CPU_CORES)
    dataset_sim_tokens = dataset.map(get_parse_function('Similar Question Tokens'), num_parallel_calls=CPU_CORES)
    dataset_dis_tokens = dataset.map(get_parse_function('Dissimilar Question Tokens'), num_parallel_calls=CPU_CORES)

    max_length = get_max_length(dataset_tokens, dataset_sim_tokens, dataset_dis_tokens)
    print("Padding using max length {}".format(max_length))
    dataset_tokens = dataset_tokens.padded_batch(BATCH_SIZE, padded_shapes=[max_length])
    dataset_sim_tokens = dataset_sim_tokens.padded_batch(BATCH_SIZE, padded_shapes=[max_length])
    dataset_dis_tokens = dataset_dis_tokens.padded_batch(BATCH_SIZE, padded_shapes=[max_length])

    dataset_zipped = tf.data.Dataset.zip((dataset_tokens, dataset_sim_tokens, dataset_dis_tokens))
    dataset_merged = dataset_zipped.map(merge_function, num_parallel_calls=CPU_CORES)

    dummy_dataset = tf.data.Dataset.from_tensor_slices(tf.constant([0], dtype=tf.int64)) \
        .repeat()

    dataset = tf.data.Dataset.zip(datasets=(dataset_merged, dummy_dataset)) \
        .shuffle(buffer_size=tf.constant(SHUFFLE_BUFFER_SIZE, dtype=tf.int64)) \
        .repeat() \
        .prefetch(PREFETCH_SIZE)

    return dataset
