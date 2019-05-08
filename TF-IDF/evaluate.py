#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate.py.py
# @Author: harry
# @Date  : 2019/5/8 下午7:05
# @Desc  : Evaluate TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from constants import *


def compute_pair(tf_idf_rst, a_id, b_id):
    return np.dot(tf_idf_rst[a_id], tf_idf_rst[b_id])


def evaluate():
    # load dataset
    with open(TEXT_FILE, 'r') as f:
        questions = f.readlines()
    with open(RELATION_FILE, 'r') as f:
        relations = f.readlines()
    # print(len(questions))
    # print(len(relations))

    # process with tf-idf
    tf_idf = TfidfVectorizer().fit_transform(questions)
    rst = tf_idf.toarray()
    # print(rst.shape)

    # evaluation
    total = 0
    diff_list = []
    for r in relations:
        r = r.split()
        anchor_id, sim_id, dis_id = int(r[0]), int(r[1]), int(r[2])

        sim_score = compute_pair(rst, anchor_id, sim_id)
        dis_score = compute_pair(rst, anchor_id, dis_id)
        diff = sim_score - dis_score

        diff_list.append(diff)
        total += 1

        print(".", end='', flush=True)

    print()
    diff_array = np.array(diff_list, dtype=float)

    pass_cnt = np.sum(diff_array >= 0.5)
    print("acc(MARGIN=0.5): {} ({} / {})".format(pass_cnt / total, pass_cnt, total))
    pass_cnt = np.sum(diff_array >= 0.4)
    print("acc(MARGIN=0.5): {} ({} / {})".format(pass_cnt / total, pass_cnt, total))
    pass_cnt = np.sum(diff_array >= 0.3)
    print("acc(MARGIN=0.5): {} ({} / {})".format(pass_cnt / total, pass_cnt, total))
    pass_cnt = np.sum(diff_array >= 0.2)
    print("acc(MARGIN=0.5): {} ({} / {})".format(pass_cnt / total, pass_cnt, total))
    pass_cnt = np.sum(diff_array >= 0.1)
    print("acc(MARGIN=0.5): {} ({} / {})".format(pass_cnt / total, pass_cnt, total))

    return


if __name__ == "__main__":
    evaluate()
