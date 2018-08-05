# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from RapGeneratorV2.data_helpers import *
from RapGeneratorV2.model import Seq2SeqModel
import math


if __name__ == '__main__':

    # 超参数
    rnn_size = 256
    num_layers = 2
    embedding_size = 256
    learning_rate = 0.001
    mode = 'train'
    use_attention = True
    beam_search = False
    beam_size = 3
    cell_type = 'LSTM'
    max_gradient_norm = 5.0
    teacher_forcing = True
    teacher_forcing_probability = 0.5

    batch_size = 32
    epochs = 40
    display = 100
    pretrained = True
    sources_txt = 'data/sources.txt'
    targets_txt = 'data/targets.txt'
    model_dir = 'model/'

    # 得到分词后的sources和targets
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射
    sources_data, targets_data, word_to_id, _ = create_dic_and_map(sources, targets)

    with tf.Session() as sess:
        model = Seq2SeqModel(
            sess=sess,
            rnn_size=rnn_size,
            num_layers=num_layers,
            embedding_size=embedding_size,
            learning_rate=learning_rate,
            word_to_id=word_to_id,
            mode=mode,
            use_attention=use_attention,
            beam_search=beam_search,
            beam_size=beam_size,
            cell_type=cell_type,
            max_gradient_norm=max_gradient_norm,
            teacher_forcing=teacher_forcing,
            teacher_forcing_probability=teacher_forcing_probability
        )

        if pretrained:
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print('Reloading model parameters..')
                model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ValueError('No such file:[{}]'.format(model_dir))
        else:
            sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            print("----- Epoch {}/{} -----".format(e + 1, epochs))
            batches = getBatches(sources_data, targets_data, batch_size)
            step = 0
            for nextBatch in batches:
                loss, summary = model.train(nextBatch)
                if step % display == 0:
                    perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                    print("----- Loss %.2f -- Perplexity %.2f" % (loss, perplexity))
                step += 1
            model.saver.save(sess, model_dir + 'seq2seq.ckpt')