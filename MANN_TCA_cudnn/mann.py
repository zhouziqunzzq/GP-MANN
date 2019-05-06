#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : mann.py
# @Author: harry
# @Date  : 2019/4/23 下午12:06
# @Desc  : MANN model goes here

import tensorflow as tf
import time
from hyper_params import *
from lstm import lstm
from TCA import TCA

tf.enable_eager_execution()


class MANNModel(object):
    def __init__(self, vocab_size, embedding_size_vocab,
                 tag_size, embedding_size_tag,
                 lstm_units, attention_units, dense_units):
        super(MANNModel, self).__init__()
        self.lstm = lstm(lstm_units=lstm_units)
        self.lstm_units = lstm_units
        self.TCA = TCA(attention_units=attention_units)
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

    @staticmethod
    def triplet_loss(anchor_pos, anchor_neg):
        # anchor_pos, anchor_neg = tf.split(y_pred, 2, axis=-1)
        loss = tf.math.maximum(0.0, MARGIN - anchor_pos + anchor_neg)
        loss = tf.math.reduce_sum(loss, axis=0)
        return loss

    def _get_state(self, tokens, tags):
        batch_size = tokens.shape[0]
        batch_seq_length = tokens.shape[1]

        w = self.embedding_vocab(tokens)
        u = self.embedding_tag(tags)
        h = tf.zeros((batch_size, self.lstm_units))

        for i in range(batch_seq_length):
            w_t = w[:, i, :]
            w_t_expanded = tf.expand_dims(w_t, axis=1)
            h_expanded = tf.expand_dims(h, axis=1)
            context, _ = self.TCA(inputs=(u, w_t_expanded, h_expanded))
            x = tf.concat([w_t, context], axis=-1)
            x_expanded = tf.expand_dims(x, axis=1)

            result = self.lstm(x_expanded)
            # result = [last_output, h, c]
            h = result[1]

        return h

    def _compute_pair(self, a_tokens, a_tags, b_tokens, b_tags):
        a_state = self._get_state(a_tokens, a_tags)
        b_state = self._get_state(b_tokens, b_tags)
        result = tf.concat([a_state, b_state], axis=-1)
        result = self.dense1(result)
        result = self.dense2(result)
        return result

    def train(self, dataset, optimizer, epochs):
        print("Start training...")
        for epoch in range(epochs):
            start = time.time()
            total_loss = 0

            for (batch, data) in enumerate(dataset):
                anchor_tokens, anchor_tags, pos_tokens, pos_tags, neg_tokens, neg_tags = data

                with tf.GradientTape() as tape:
                    anchor_pos = self._compute_pair(anchor_tokens, anchor_tags,
                                                    pos_tokens, pos_tags)
                    anchor_neg = self._compute_pair(anchor_tokens, anchor_tags,
                                                    neg_tokens, neg_tags)

                    batch_loss = self.triplet_loss(anchor_pos, anchor_neg)

                total_loss += batch_loss

                variables = self.embedding_vocab.variables + self.embedding_tag.variables + \
                            self.lstm.variables + self.TCA.variables + \
                            self.dense1.variables + self.dense2.variables
                gradients = tape.gradient(batch_loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))

                print(".", end='', flush=True)
                if batch % 10 == 0:
                    print()
                    print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()[0])
                          )

            end = time.time()
            print("Epoch {} cost {}".format(epoch + 1, end - start))

        print("End training...")
