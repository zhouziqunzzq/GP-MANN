#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : TCA.py
# @Author: harry
# @Date  : 2019/5/6 下午2:12
# @Desc  : TCA

import tensorflow as tf


class TCA(tf.keras.Model):
    def __init__(self, attention_units):
        super(TCA, self).__init__()
        self.W_ac = tf.keras.layers.Dense(
            units=attention_units,
            activation=tf.keras.activations.tanh,
            use_bias=False,
        )
        self.V_ac = tf.keras.layers.Dense(
            units=1,
            activation=None,
            use_bias=False,
        )

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tuple)
        u, w_t, h_tm1 = inputs

        w_t = tf.tile(w_t, multiples=[1, u.shape[1], 1])
        h_tm1 = tf.tile(h_tm1, multiples=[1, u.shape[1], 1])
        x = tf.concat([u, w_t, h_tm1], axis=-1)
        x = self.W_ac(x)
        scores = self.V_ac(x)
        scores = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(scores * u, axis=1)

        return context, scores
