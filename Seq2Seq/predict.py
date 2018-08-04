# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np
from data_helpers import *
from model import Seq2SeqModel
import sys


def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    '''
    predicted_ids = np.squeeze(np.array(predict_ids))
    predict_seq = [id2word[single_predict] for single_predict in predicted_ids]
    print(''.join(predict_seq))


if __name__ == '__main__':

    # 超参数
    rnn_size = 1024
    num_layers = 2
    embedding_size = 1024
    batch_size = 128
    learning_rate = 0.0001
    epochs = 100
    steps_per_checkpoint = 5
    filepath = 'data/data.txt'
    model_dir = 'model/'

    # 加载并预处理数据
    data = load_data(filepath)
    processed_data, word_to_id, id_to_word = process_all_data(data)

    with tf.Session() as sess:
        model = Seq2SeqModel(rnn_size, num_layers, embedding_size, learning_rate, word_to_id,
                             mode='decode', use_attention=True, beam_search=False, beam_size=5, max_gradient_norm=5.0)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(model_dir))
        # model.saver.restore(sess, model_dir)
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            batch = sentence2enco(sentence, word_to_id)
            predicted_ids = model.infer(sess, batch)
            predict_ids_to_seq(predicted_ids, id_to_word, 8)
            print("> ", "")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
