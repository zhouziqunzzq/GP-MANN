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
            return_sequences=True,
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
            return result[1], result[0]

        def _compute_sim_attention(h):
            h1, h2 = h

            a_norm = tf.sqrt(tf.reduce_sum(tf.square(h1), axis=-1))
            a_norm = tf.expand_dims(a_norm, axis=-1)
            b_norm = tf.sqrt(tf.reduce_sum(tf.square(h2), axis=-1))
            b_norm = tf.expand_dims(b_norm, axis=-1)

            norm_matrix = tf.tensordot(a_norm, tf.transpose(b_norm), [[1], [0]])

            dot_matrix = tf.tensordot(h1, tf.transpose(h2), [[1], [0]])

            sim = dot_matrix / norm_matrix
            return sim

        def _compute_pair(a, b):
            r_a, h_a = _get_state(a)
            r_b, h_b = _get_state(b)
            # r_a.shape == (batch_size, lstm_units)
            # h_a.shape == (batch_size, seq_a_length, lstm_units)
            # r_b.shape == (batch_size, lstm_units)
            # h_b.shape == (batch_size, seq_b_length, lstm_units)

            # Similarity Attention
            # iterate over batch
            sim_attention_matrix = tf.map_fn(
                _compute_sim_attention,
                elems=(h_a, h_b),
                dtype=tf.float32
            )
            # sim_attention_matrix == (batch_size, seq_a_length, seq_b_length)

            s_a = tf.reduce_sum(sim_attention_matrix, axis=2)
            s_b = tf.reduce_sum(sim_attention_matrix, axis=1)
            # s_a.shape == (batch_size, seq_a_length)
            # s_b.shape == (batch_size, seq_b_length)

            att_a_weight = sim_attention_matrix[:, :, -1]
            att_a_weight = tf.expand_dims(att_a_weight, axis=-1)
            h_att_a = att_a_weight * h_a
            h_att_a = tf.reduce_sum(h_att_a, axis=1)
            # h_att_a.shape == (batch_size, lstm_units)

            att_b_weight = sim_attention_matrix[:, -1, :]
            att_b_weight = tf.expand_dims(att_b_weight, axis=-1)
            h_att_b = att_b_weight * h_b
            h_att_b = tf.reduce_sum(h_att_b, axis=1)
            # h_att_b.shape == (batch_size, lstm_units)

            # Similarity Score Layer
            result = tf.keras.layers.concatenate(
                inputs=[r_a, r_b, s_a, s_b, h_att_a, h_att_b],
                axis=-1,
            )
            result = self.dense1(result)
            result = self.dense2(result)
            return result, sim_attention_matrix

        training = self.training

        if training:
            print("Building MANN for training...")
            # build training model with triplet loss
            # unpack inputs (expecting 3 tensors in training mode)
            anchor_tokens, pos_tokens, neg_tokens = inputs
            # compute MANN(anchor, pos)
            anchor_pos, _ = _compute_pair(anchor_tokens, pos_tokens)
            # compute MANN(anchor, neg)
            anchor_neg, _ = _compute_pair(anchor_tokens, neg_tokens)
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
