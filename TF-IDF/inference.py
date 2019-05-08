#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : inference.py.py
# @Author: harry
# @Date  : 2019/4/23 下午12:13
# @Desc  : MANN inference with eager execution enabled

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from constants import *
from evaluate import compute_pair
from utils import *


def main():
    # load dataset
    with open(TEXT_FILE, 'r') as f:
        questions = f.readlines()
    while True:
        try:
            # input data
            e1_text = clean_empty_lines(clean_html(input("Input exercise 1: ")))
            e2_text = clean_empty_lines(clean_html(input("Input exercise 2: ")))

            # construct tf-idf vectorizer
            documents = questions.copy()
            documents.append(e1_text)
            documents.append(e2_text)
            tf_idf = TfidfVectorizer().fit_transform(documents)
            rst = tf_idf.toarray()

            score = compute_pair(rst, -2, -1)
            # print(repr(sim_score))
            print("Similar score: {}".format(score))
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    main()
