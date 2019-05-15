#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_cosine_sim.py
# @Author: harry
# @Date  : 2019/5/15 下午9:55
# @Desc  : Test cosine similarity

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

a = np.array([[1, 2, 3],
              [1, 3, 5]], dtype=float)
b = np.array([[2, 4, 6],
              [2, 4, 6]], dtype=float)

a_norm = tf.sqrt(tf.reduce_sum(tf.square(a), axis=-1))
a_norm = tf.expand_dims(a_norm, axis=-1)
b_norm = tf.sqrt(tf.reduce_sum(tf.square(b), axis=-1))
b_norm = tf.expand_dims(b_norm, axis=-1)

norm_matrix = tf.tensordot(a_norm, tf.transpose(b_norm), [[1], [0]])
# print(repr(norm_matrix))

dot_matrix = tf.tensordot(a, tf.transpose(b), [[1], [0]])
# print(repr(dot_matrix))

sim = dot_matrix / norm_matrix
# sim.shape == (seq_length_a, seq_length_b)
print(repr(sim))

s_a = tf.reduce_sum(sim, axis=1)
s_b = tf.reduce_sum(sim, axis=0)
print(repr(s_a))
print(repr(s_b))

h_att_a = sim[:, -1]
print(repr(h_att_a))
h_att_b = sim[-1, :]
print(repr(h_att_b))
