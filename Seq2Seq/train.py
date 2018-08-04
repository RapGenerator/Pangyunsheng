# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from data_helpers import *
from model import Seq2SeqModel
import math

if __name__ == '__main__':

    # 超参数
    rnn_size = 1024
    num_layers = 2
    embedding_size = 1024
    batch_size = 128
    learning_rate = 0.0001
    epochs = 200
    filepath = 'data/data.txt'

    # 加载并预处理数据
    data = load_data(filepath)
    processed_data, word_to_id, _ = process_all_data(data)

    with tf.Session() as sess:
        model = Seq2SeqModel(rnn_size, num_layers, embedding_size, learning_rate, word_to_id,
                             mode='train', use_attention=True, beam_search=False, beam_size=5, max_gradient_norm=5.0)
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            print("----- Epoch {}/{} -----".format(e + 1, epochs))
            batches = getBatches(processed_data, batch_size)
            for nextBatch in batches:
                loss, summary = model.train(sess, nextBatch)
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print("----- Loss %.2f -- Perplexity %.2f" % (loss, perplexity))
                model.saver.save(sess, 'model/seq2seq.ckpt')
