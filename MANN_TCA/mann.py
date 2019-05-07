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
            return_sequences=False,
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
            return result[1]

        def _compute_pair(a, tags_a, b, tags_b):
            a_state = _get_state(a, self.embedding_tag(tags_a))
            b_state = _get_state(b, self.embedding_tag(tags_b))
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
            # unpack inputs (expecting 6 tensors in training mode)
            anchor_tokens, anchor_tags, pos_tokens, pos_tags, neg_tokens, neg_tags = inputs
            # compute MANN(anchor, pos)
            anchor_pos = _compute_pair(anchor_tokens, anchor_tags, pos_tokens, pos_tags)
            # compute MANN(anchor, neg)
            anchor_neg = _compute_pair(anchor_tokens, anchor_tags, neg_tokens, neg_tags)
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
