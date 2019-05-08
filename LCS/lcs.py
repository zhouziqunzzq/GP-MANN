#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : lcs.py
# @Author: harry
# @Date  : 2019/5/8 上午10:41
# @Desc  : Longest Common SubSequence

import time


def lcs(x, y):
    matrix = [list()] * (len(x) + 1)
    for index_x in range(len(matrix)):
        matrix[index_x] = [list()] * (len(y) + 1)

    for index_x in range(1, len(x) + 1):
        for index_y in range(1, len(y) + 1):
            if x[index_x - 1] == y[index_y - 1]:  # 这里利用属性一
                matrix[index_x][index_y] = matrix[index_x - 1][index_y - 1] + [x[index_x - 1]]
            elif len(matrix[index_x][index_y - 1]) > len(matrix[index_x - 1][index_y]):  # 这里和下面利用属性二
                matrix[index_x][index_y] = matrix[index_x][index_y - 1]
            else:
                matrix[index_x][index_y] = matrix[index_x - 1][index_y]

    return matrix[len(x)][len(y)]


def compute_pair(x, y):
    return len(lcs(x, y)) / max(len(x), len(y))


if __name__ == '__main__':
    a = [i for i in range(256)]
    b = [i for i in range(256) if i % 2 == 0]
    start = time.time()
    rst = lcs(a, b)
    print(rst)
    print(len(rst))
    print(time.time() - start)
