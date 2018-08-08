#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : CreateDictionary.py
# @Author: harry
# @Date  : 18-8-6 下午5:37
# @Desc  : Create dictionary.txt

from SkipThought.DataHelpers import *
from SkipThought.HyperParameter import HyperParameter

if __name__ == '__main__':
    # Load data
    hp = HyperParameter()
    data = load_and_cut_data('data/data.txt')

    # Create dictionary in mem
    word_dic_new = list(set([character for line in data for character in line]))

    # Save to file
    with open(hp.dictionary_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(word_dic_new))

    print("Dictionary saved in {}".format(hp.dictionary_txt))
