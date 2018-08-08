# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np

from SkipThought.DataHelpers import *
from SkipThought.model import SkipThoughtModel
from SkipThought.HyperParameter import HyperParameter
import sys


def predict_ids_to_seq(predict_ids, id2word, beam_size):
    """
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    """
    for single_predict in predict_ids:
        for i in range(beam_size):
            print("Beam search result {}：".format(i + 1))
            predict_list = np.ndarray.tolist(single_predict[:, i])
            predict_seq = [id2word[idx] for idx in predict_list]
            print(" ".join(predict_seq))
            print()


if __name__ == '__main__':
    # 超参数
    hp = HyperParameter()
    rnn_size = hp.rnn_size
    num_layers = hp.num_layers
    embedding_size = hp.embedding_size
    batch_size = hp.batch_size
    learning_rate = hp.learning_rate
    epochs = hp.epochs
    steps_per_checkpoint = hp.steps_per_checkpoint
    data_txt = hp.data_txt
    model_dir = hp.model_dir
    beam_size = hp.beam_size
    writer = None

    # 得到分词后的sources和targets
    data = load_and_cut_data(data_txt)

    # 根据sources和targets创建词典，并映射
    data, word_to_id, id_to_word = create_dic_and_map(data)

    with tf.Session() as sess:
        model = SkipThoughtModel(
            sess=sess,
            rnn_size=rnn_size,
            num_layers=num_layers,
            embedding_size=embedding_size,
            word_to_id=word_to_id,
            mode='predict',
            learning_rate=learning_rate,
            use_attention=True,
            beam_search=True,
            beam_size=beam_size,
            cell_type='LSTM',
            max_gradient_norm=5.0,
            writer=writer,
        )
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(model_dir))

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            batch = sentence2enco(sentence, word_to_id)
            predict_pre, predict_post = model.infer(batch)

            predict_ids_to_seq(predict_pre, id_to_word, beam_size)
            predict_ids_to_seq(predict_post, id_to_word, beam_size)
            sys.stdout.flush()
            sentence = input("> ")
