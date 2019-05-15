#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_tfrecord.py
# @Author: harry
# @Date  : 2019/4/8 下午4:24
# @Desc  : Just a simple toy to learn how to use TFRecord

import tensorflow as tf

tf.enable_eager_execution()

# create Dataset from TFRecords
filenames = ["./tf_data/leetcode_pairwise.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames=filenames)
print(dataset)


# define parse functions
def _convert_sparse(features: dict, entry_list: list):
    for k in entry_list:
        features[k] = tf.sparse.to_dense(features[k])


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
        _convert_sparse(parsed_features, [
            'Tokens',
            'Tags',
            'Similar Question Tokens',
            'Similar Question Tags',
            'Dissimilar Question Tokens',
            'Dissimilar Question Tags',
        ])
        return parsed_features[feature_name]

    return _parse_function


def merge_function(*args):
    return args


# parse Dataset
dataset_tokens = dataset.map(get_parse_function('Tokens'))
dataset_sim_tokens = dataset.map(get_parse_function('Similar Question Tokens'))
dataset_dis_tokens = dataset.map(get_parse_function('Dissimilar Question Tokens'))
# pad & batch
dataset_tokens = dataset_tokens.padded_batch(2, padded_shapes=[None])
dataset_sim_tokens = dataset_sim_tokens.padded_batch(2, padded_shapes=[None])
dataset_dis_tokens = dataset_dis_tokens.padded_batch(2, padded_shapes=[None])
# zip
dataset_final = tf.data.Dataset.zip((dataset_tokens, dataset_sim_tokens, dataset_dis_tokens))
dataset_final = dataset_final.map(merge_function)

# iterate over Dataset
for x in dataset_final.take(1):
    print(repr(x))

# === Another way to read TFRecord files ===
# Note that this method is deprecated
# because it's pure python_io method
# without any optimization compared to
# tf.data.TFRecordDataset.
# ==========================================
# record_iterator = tf.python_io.tf_record_iterator(path="./tf_data/leetcode_pairwise.tfrecord")
# for string_record in record_iterator:
#     example = tf.train.Example()
#     example.ParseFromString(string_record)
#     print(example)
#     # Exit after 1 iteration as this is purely demonstrative.
#     break
