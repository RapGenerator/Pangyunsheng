# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np
from RapGenerator.data_helpers import *
from RapGenerator.model import Seq2SeqModel
import sys


def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    '''
    for single_predict in predict_ids:
        for i in range(beam_szie):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = [id2word[idx] for idx in predict_list[0]]
            print(" ".join(predict_seq))


if __name__ == '__main__':

    # 超参数
    rnn_size = 1024
    num_layers = 2
    embedding_size = 1024
    batch_size = 128
    learning_rate = 0.0001
    epochs = 100
    steps_per_checkpoint = 5
    sources_txt = 'data/sources.txt'
    targets_txt = 'data/targets.txt'
    model_dir = 'model/'

    # 得到分词后的sources和targets
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射
    sources_data, targets_data, word_to_id, id_to_word = create_dic_and_map(sources, targets)

    with tf.Session() as sess:
        model = Seq2SeqModel(rnn_size, num_layers, embedding_size, learning_rate, word_to_id, mode='decode',
                             use_attention=True, beam_search=True, beam_size=5, cell_type='LSTM', max_gradient_norm=5.0)
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
            predict_ids_to_seq(predicted_ids, id_to_word, 5)
            print("> ", "")
            sys.stdout.flush()
            sentence = sys.stdin.readline()