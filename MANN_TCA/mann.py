#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : mann.py
# @Author: harry
# @Date  : 2019/4/23 下午12:06
# @Desc  : MANN model goes here

import tensorflow as tf
import numpy as np
from hyper_params import *
from LSTMCell_with_TCA import LSTMCellWithTCA


class MANNModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size_vocab,
                 tag_size, embedding_size_tag,
                 lstm_units, dense_units, training=True):
        super(MANNModel, self).__init__(name='MANN_model')
        self.training = training
        self.embedding_vocab = tf.keras.layers.Embedding(
            name="embedding_vocab",
            input_dim=vocab_size,
            output_dim=embedding_size_vocab,
        )
        self.embedding_tag = tf.keras.layers.Embedding(
            name="embedding_tag",
            input_dim=tag_size,
            output_dim=embedding_size_tag,
        )
        self.lstm_cell = LSTMCellWithTCA(
            name="lstm_cell_with_TCA",
            units=lstm_units,
            attention_units=ATTENTION_UNITS,
            feature_dim=EMBEDDING_SIZE_TAG,
            kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
        )
        self.lstm = tf.keras.layers.RNN(
            name="lstm_with_TCA",
            cell=self.lstm_cell,
            return_sequences=True,
            return_state=True,
        )
        # self.lstm = tf.keras.layers.LSTM(
        #     name="lstm",
        #     units=lstm_units,
        #     return_sequences=False,
        #     return_state=True,
        #     # unroll=False,
        #     kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
        #     # dropout=LSTM_DROPOUT,
        # )
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
        def _get_state(x, tags_emb):
            result = self.embedding_vocab(x)
            result = self.lstm(
                result,
                constants=[tags_emb]
            )
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

        def _compute_pair(a, tags_a, b, tags_b):
            r_a, h_a = _get_state(a, self.embedding_tag(tags_a))
            r_b, h_b = _get_state(b, self.embedding_tag(tags_b))
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

            # Similarity Score Layer
            result = tf.keras.layers.concatenate(
                inputs=[r_a, r_b],
                axis=-1,
            )
            result = self.dense1(result)
            result = self.dense2(result)
            return result, sim_attention_matrix

        training = self.training

        if training:
            print("Building MANN for training...")
            # build training model with triplet loss
            # unpack inputs (expecting 6 tensors in training mode)
            anchor_tokens, anchor_tags, pos_tokens, pos_tags, neg_tokens, neg_tags = inputs
            # compute MANN(anchor, pos)
            anchor_pos, _ = _compute_pair(anchor_tokens, anchor_tags, pos_tokens, pos_tags)
            # compute MANN(anchor, neg)
            anchor_neg, _ = _compute_pair(anchor_tokens, anchor_tags, neg_tokens, neg_tags)
            # return concatenated results for loss computation
            return tf.keras.layers.concatenate(
                inputs=[anchor_pos, anchor_neg],
                axis=-1,
            )
        else:
            print("Building MANN for inference...")
            # build inference model
            # unpack inputs (expecting 4 tensors in inference mode)
            a_tokens, a_tags, b_tokens, b_tags = inputs
            return _compute_pair(a_tokens, a_tags, b_tokens, b_tags)
