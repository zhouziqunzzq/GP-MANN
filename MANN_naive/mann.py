#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : mann.py
# @Author: harry
# @Date  : 2019/4/23 下午12:06
# @Desc  : MANN model goes here

import tensorflow as tf
from hyper_params import *


class MANNModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_units, dense_units, training=True):
        super(MANNModel, self).__init__(name='MANN_model')
        self.training = training
        self.embedding = tf.keras.layers.Embedding(
            name="embedding",
            input_dim=vocab_size,
            output_dim=embedding_size,
        )
        self.lstm = tf.keras.layers.CuDNNLSTM(
            name="lstm",
            units=lstm_units,
            return_sequences=False,
            return_state=True,
            # unroll=False,
            kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
            # dropout=LSTM_DROPOUT,
        )
        self.dense1 = tf.keras.layers.Dense(
            name="dense_1",
            units=dense_units,
            activation=tf.keras.activations.relu,
            kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
        )
        self.dense2 = tf.keras.layers.Dense(
            name="dense_2",
            units=1,
            activation=tf.keras.activations.sigmoid,
            kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
        )

    # TODO: how to use this "training" arg ???
    def call(self, inputs, training=None, mask=None):
        def _get_state(x):
            result = self.embedding(x)
            result = self.lstm(result)
            return result[1]

        def _compute_pair(a, b):
            a_state = _get_state(a)
            b_state = _get_state(b)
            result = tf.keras.layers.concatenate(
                inputs=[a_state, b_state],
                axis=-1,
            )
            result = self.dense1(result)
            result = self.dense2(result)
            return result

        training = self.training

        if training:
            print("Building MANN for training...")
            # build training model with triplet loss
            # unpack inputs (expecting 3 tensors in training mode)
            anchor_tokens, pos_tokens, neg_tokens = inputs
            # compute MANN(anchor, pos)
            anchor_pos = _compute_pair(anchor_tokens, pos_tokens)
            # compute MANN(anchor, neg)
            anchor_neg = _compute_pair(anchor_tokens, neg_tokens)
            # return concatenated results for loss computation
            return tf.keras.layers.concatenate(
                inputs=[anchor_pos, anchor_neg],
                axis=-1,
            )
        else:
            print("Building MANN for inference...")
            # build inference model
            # unpack inputs (expecting 2 tensors in inference mode)
            a_tokens, b_tokens = inputs
            return _compute_pair(a_tokens, b_tokens)
