#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: harry
# @Date  : 2019/4/23 下午12:40
# @Desc  : Utils

import lxml.html
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决中文乱码


def clean_html(raw: str) -> str:
    document = lxml.html.document_fromstring(raw)
    return document.text_content()


def clean_empty_lines(raw: str) -> str:
    return ' '.join([line.strip() for line in raw.split('\n') if line.strip() != ''])


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
