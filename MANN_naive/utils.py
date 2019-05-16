#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: harry
# @Date  : 2019/4/23 下午12:40
# @Desc  : Utils

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import lxml.html
from constants import *
from tokenizer import Tokenizer


def convert_sparse(features: dict, entry_list: list):
    for k in entry_list:
        features[k] = tf.sparse.to_dense(features[k])


def tf_bytes_feature(value: list) -> tf.train.Feature:
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value)
    )


def tf_int64_feature(value: list) -> tf.train.Feature:
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=value)
    )


def get_pair_example(a: dict, b: dict) -> tf.train.Example:
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'Tokens 1': tf_int64_feature(a['tokens']),
                'Tokens 2': tf_int64_feature(b['tokens']),
            }
        )
    )


def load_word2id() -> (dict, dict):
    word2id = {}
    id2word = {}
    wid = 0
    for w in open(VOCAB_PATH).readlines():
        word2id[w.strip()] = wid
        id2word[wid] = w.strip()
        wid += 1
    return word2id, id2word


def clean_html(raw: str) -> str:
    document = lxml.html.document_fromstring(raw)
    return document.text_content()


def clean_empty_lines(raw: str) -> str:
    return ' '.join([line.strip() for line in raw.split('\n') if line.strip() != ''])


def tokenize_raw_text(raw_text: str) -> list:
    tokenizer = Tokenizer(clean_empty_lines(clean_html(raw_text)))
    return tokenizer.tokenize()


def tokenize_raw_text_to_id(word2id: dict, raw_text: str) -> list:
    # we assume that self.word2id is valid
    assert len(word2id) > 0

    tokens = []
    for t in tokenize_raw_text(raw_text=raw_text):
        t = t.lower()
        if t in word2id:
            tokens.append(word2id[t])
        else:
            tokens.append(word2id[UNK_TOKEN])

    return tokens


def plot_attention(attention, s1, s2):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    psm = ax.matshow(attention, cmap='viridis')

    fig.colorbar(psm)

    font_dict = {'fontsize': 14}

    ax.set_yticklabels([''] + s1, fontdict=font_dict)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticklabels([''] + s2, fontdict=font_dict, rotation=90)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
