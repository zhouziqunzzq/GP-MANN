#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: harry
# @Date  : 2019/4/23 下午12:40
# @Desc  : Utils

import lxml.html
from constants import *
from tokenizer import Tokenizer


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
