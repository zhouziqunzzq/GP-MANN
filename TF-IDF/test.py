#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py.py
# @Author: harry
# @Date  : 2019/5/8 下午4:21
# @Desc  : Just a test

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

data = [
    'Hello this is Harry',
    'Hi this is Kate',
    'Harry is eating an apple',
    'Kate is running',
    'Hello this is Harry',
]

tf_idf = TfidfVectorizer().fit_transform(data)

rst = tf_idf.toarray()
print(rst[0])
print(rst[3])
print(np.dot(rst[0], rst[1]))
print(np.dot(rst[0], rst[2]))
print(np.dot(rst[0], rst[3]))
print(np.dot(rst[0], rst[4]))
