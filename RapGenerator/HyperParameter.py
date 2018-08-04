#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : HyperParameter.py
# @Author: harry
# @Date  : 18-8-4 下午2:24
# @Desc  : HyperParameter wrapper class


class HyperParameter(object):
    def __init__(self):
        # 超参数
        self.rnn_size = 64
        self.num_layers = 2
        self.embedding_size = 64
        self.batch_size = 128
        self.learning_rate = 0.001
        self.epochs = 100
        self.steps_per_checkpoint = 5
        self.sources_txt = 'data/sources.txt'
        self.targets_txt = 'data/targets.txt'
        self.model_dir = 'model/'
        self.print_loss_steps = 100
