# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import csv

data_file = 'data/df_all_pinyin_clear.csv'
new_data_file = 'data/data.txt'
writer = open(new_data_file, 'w', encoding='utf-8')


with open(data_file, 'r', encoding='utf-8') as f:
    lines = csv.reader(f)

    for line in lines:
        writer.writelines(line[0] + '\n')