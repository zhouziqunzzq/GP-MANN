#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : tokenizer.py
# @Author: harry
# @Date  : 2019/4/4 下午12:31
# @Desc  : Yet another tokenizer for leetcode data


class Tokenizer(object):
    def __init__(self, raw_str: str):
        self.raw_str = raw_str

    def tokenize(self) -> list:
        # init
        buffer = ""
        last_space = False
        tokens = []

        # loop over raw_str
        for c in self.raw_str:
            if c.isalpha():
                buffer += c
                last_space = False
            elif c.isspace():
                if last_space:
                    continue
                else:
                    if len(buffer) > 0:
                        tokens.append(buffer)
                    buffer = ""
                last_space = True
            else:
                # special chars
                if len(buffer) > 0:
                    tokens.append(buffer)
                tokens.append(c)
                buffer = ""
                last_space = False

        # post process
        if len(buffer) > 0:
            tokens.append(buffer)

        return tokens


if __name__ == "__main__":
    tokenizer = Tokenizer("    ")
    print(repr(tokenizer.tokenize()))
