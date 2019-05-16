#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : LSTMCell_with_TCA.py
# @Author: harry
# @Date  : 2019/5/1 下午4:06
# @Desc  : LSTM Cell with TCA(Text-Concept Attention)

import tensorflow as tf


# Note: we subclass from LSTMCell for now,
# so we cannot use CuDNN acceleration...
class LSTMCellWithTCA(tf.keras.layers.LSTMCell):
    def __init__(self,
                 units,
                 attention_units,
                 feature_dim,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(LSTMCellWithTCA, self).__init__(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            **kwargs,
        )

        # Attention related params
        self.feature_dim = feature_dim
        self.attention_units = attention_units

        # variables to be built in build()
        self.W_ac = None
        self.V_ac = None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.W_ac = self.add_weight(
            shape=(self.feature_dim + input_dim + self.units, self.attention_units),
            name='W_ac',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )
        self.V_ac = self.add_weight(
            shape=(self.attention_units, 1),
            name='V_ac',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        # need to change input_shape for LSTMCell super class here!
        input_shape = (input_shape[0], self.feature_dim + input_dim)

        super(LSTMCellWithTCA, self).build(input_shape)

    # Note: constants are used to pass features to Attention
    def call(self, inputs, states, training=None, constants=None):
        u = constants[0]
        w_t = tf.expand_dims(inputs, axis=1)
        h_tm1 = tf.expand_dims(states[0], axis=1)

        # TCA(Text-Concept Attention)
        w_t = tf.tile(w_t, multiples=[1, u.shape[1], 1])
        h_tm1 = tf.tile(h_tm1, multiples=[1, u.shape[1], 1])
        x = tf.concat([u, w_t, h_tm1], axis=-1)
        x = tf.tensordot(x, self.W_ac, axes=[[2], [0]])
        x = tf.tanh(x)
        scores = tf.tensordot(x, self.V_ac, axes=[[2], [0]])
        scores = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(scores * u, axis=1)

        concat_inputs = tf.concat([inputs, context], axis=-1)
        # print(repr(concat_inputs))

        return super(LSTMCellWithTCA, self).call(concat_inputs, states, training)

    def get_config(self):
        base_config = super(LSTMCellWithTCA, self).get_config()
        base_config['feature_dim'] = self.feature_dim
        base_config['attention_units'] = self.attention_units
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
