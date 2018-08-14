#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : CreateDictionary.py
# @Author: harry
# @Date  : 18-8-6 下午5:37
# @Desc  : Create dictionary.txt

from MTA_LSTM.DataHelpers import *

if __name__ == '__main__':
    # Load data
    context, keywords = load_and_cut_data('data/composition.txt')

    # Create dictionary in mem
    word_dic_new = list(set([character for line in (context + keywords) for character in line]))

    # Save to file
    with open('data/dictionary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(word_dic_new))

    print("Dictionary saved in {}".format('data/dictionary.txt'))
