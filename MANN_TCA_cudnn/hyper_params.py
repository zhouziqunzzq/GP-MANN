#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : hyper_params.py
# @Author: harry
# @Date  : 2019/4/23 下午12:15
# @Desc  : Hyper parameters definition

EMBEDDING_SIZE_VOCAB = 100
EMBEDDING_SIZE_TAG = 100
LSTM_UNITS = 128
ATTENTION_UNITS = 128
DENSE_UNITS = 256
LEARNING_RATE = 0.001

# Gradient Clipping is IMPORTANT!!!!!!
CLIP_NORM = 1.0

MARGIN = 0.5
REG_LAMBDA = 0.00004
