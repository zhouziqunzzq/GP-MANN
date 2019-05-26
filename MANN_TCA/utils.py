#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: harry
# @Date  : 2019/4/23 下午12:40
# @Desc  : Utils

import tensorflow as tf
import matplotlib.ticker as ticker
import lxml.html
from constants import *
from tokenizer import Tokenizer
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决中文乱码


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


def load_tag2id() -> dict:
    tag2id = {}
    tid = 0
    for t in open(TAG_PATH).readlines():
        tag2id[t.strip()] = tid
        tid += 1
    return tag2id


def pad_tags(tag2id: dict, a: list, b: list) -> (list, list):
    if len(a) == len(b):
        return a, b
    # assume len(a) > len(b), so just pad b
    swapped = False
    if len(a) < len(b):
        a, b = b, a
        swapped = True
    b = b + ([tag2id[PAD_TOKEN]] * (len(a) - len(b)))
    if swapped:
        return b, a
    else:
        return a, b


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


def tokenize_raw_tags_to_id(tag2id: dict, raw_tags: str) -> list:
    assert len(tag2id) > 0

    tags = []
    for t in raw_tags.split():
        t = t.lower()
        if t in tag2id:
            tags.append(tag2id[t])
        else:
            tags.append(tag2id[PAD_TOKEN])

    return tags


def plot_attention(attention, s1, s2):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    psm = ax.matshow(attention, cmap='gist_gray')

    fig.colorbar(psm)

    font_dict = {'fontsize': 14}

    ax.set_yticklabels([''] + s1, fontdict=font_dict)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticklabels([''] + s2, fontdict=font_dict, rotation=90)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def plot_pie(labels: list, sizes: list, colors: list, explode: list):
    plt.figure()  # 调节图形大小
    labels = labels  # 定义标签
    sizes = sizes  # 每块值
    colors = colors  # 每块颜色定义 ['red', 'yellowgreen', 'lightskyblue', 'yellow']
    explode = explode  # 将某一块分割出来，值越大分割出的间隙越大
    patches, text1, text2 = plt.pie(sizes,
                                    explode=explode,
                                    labels=labels,
                                    colors=colors,
                                    autopct='%3.2f%%',  # 数值保留固定小数位
                                    shadow=False,  # 无阴影设置
                                    startangle=90,  # 逆时针起始角度设置
                                    pctdistance=0.6)  # 数值距圆心半径倍数距离
    # patches饼图的返回值，text1饼图外label的文本，text2饼图内部的文本
    for t in text2:
        t.set_color('white')
    # x，y轴刻度设置一致，保证饼图为圆形
    plt.axis('equal')
    return plt
