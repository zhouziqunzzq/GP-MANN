#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: harry
# @Date  : 2019/4/23 下午12:40
# @Desc  : Utils

import tensorflow as tf
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


def load_word2id() -> dict:
    word2id = {}
    wid = 0
    for w in open(VOCAB_PATH).readlines():
        word2id[w.strip()] = wid
        wid += 1
    return word2id


def clean_html(raw: str) -> str:
    document = lxml.html.document_fromstring(raw)
    return document.text_content()


def clean_empty_lines(raw: str) -> str:
    return ' '.join([line.strip() for line in raw.split('\n') if line.strip() != ''])


def tokenize_raw_text(raw_text: str) -> list:
    tokenizer = Tokenizer(clean_empty_lines(clean_html(raw_text)))
    return tokenizer.tokenize()


def tokenize_raw_text_to_id(word2id: dict, raw_text: str) -> list:
    # we assume that word2id is valid
    assert len(word2id) > 0

    tokens = []
    for t in tokenize_raw_text(raw_text=raw_text):
        t = t.lower()
        if t in word2id:
            tokens.append(word2id[t])
        else:
            tokens.append(word2id[UNK_TOKEN])

    return tokens


def pad_tokens(word2id: dict, tokens: list) -> list:
    # we assume that word2id is valid
    assert len(word2id) > 0

    if len(tokens) > SEQ_LENGTH:
        raise Exception("token length is bigger than SEQ_LENGTH")
    else:
        return tokens + ([word2id[PAD_TOKEN]] * (SEQ_LENGTH - len(tokens)))
