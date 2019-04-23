#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: harry
# @Date  : 2019/4/23 下午12:40
# @Desc  : Utils

import tensorflow as tf


def convert_sparse(features: dict, entry_list: list):
    for k in entry_list:
        features[k] = tf.sparse.to_dense(features[k])


def tf_bytes_feature(value: list) -> tf.train.Feature:
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value)
    )


def tf_int64_feature(value: list) -> tf.train.Feature:
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=value)
    )


def get_pair_example(a: dict, b: dict) -> tf.train.Example:
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'Tokens 1': tf_int64_feature(a['tokens']),
                'Tokens 2': tf_int64_feature(b['tokens']),
            }
        )
    )
