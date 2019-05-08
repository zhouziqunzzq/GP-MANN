#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : inference.py.py
# @Author: harry
# @Date  : 2019/4/23 下午12:13
# @Desc  : MANN inference with eager execution enabled

from lcs import lcs, compute_pair
from utils import *


def main():
    # load word2id
    word2id = load_word2id()

    while True:
        try:
            # input data
            e1_text = input("Input exercise 1: ")
            e2_text = input("Input exercise 2: ")

            # tokenize
            e1_tokens = tokenize_raw_text_to_id(word2id, e1_text)
            e2_tokens = tokenize_raw_text_to_id(word2id, e2_text)

            print(e1_tokens)
            print(e2_tokens)

            score = compute_pair(e1_tokens, e2_tokens)
            # print(repr(sim_score))
            print("Similar score: {}".format(score))
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    main()
