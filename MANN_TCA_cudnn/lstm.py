#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : lstm.py
# @Author: harry
# @Date  : 2019/5/6 下午2:13
# @Desc  : LSTM wrapper

import tensorflow as tf
from hyper_params import *


def lstm(lstm_units: int):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(
            name="lstm",
            units=lstm_units,
            return_sequences=False,
            return_state=True,
            kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
        )
    else:
        return tf.keras.layers.LSTM(
            name="lstm",
            units=lstm_units,
            return_sequences=False,
            return_state=True,
            kernel_regularizer=tf.keras.regularizers.l2(REG_LAMBDA),
        )
