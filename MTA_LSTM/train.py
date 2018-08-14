# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from MTA_LSTM.DataHelpers import *
from MTA_LSTM.model import MTALSTM
from MTA_LSTM.HyperParameter import HyperParameter
import math
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':

    hp = HyperParameter()
    context, keywords = load_and_cut_data(hp.data_txt)
    context, keywords, word_to_id, id_to_word = create_dic_and_map(context, keywords)

    # Train
    with tf.Graph().as_default(), tf.Session() as sess:
        # Build model
        model = MTALSTM(
            sess=sess,
            num_layers=hp.num_layers,
            num_steps=hp.num_steps,
            rnn_size=hp.rnn_size,
            embedding_size=hp.embedding_size,
            word_to_id=word_to_id,
            learning_rate=hp.learning_rate,
            num_keywords=hp.num_keywords,
            max_gradient_norm=5.0,
            is_training=True,
            max_to_keep=hp.max_to_keep,
        )

        # Trying to restore model
        ckpt = tf.train.get_checkpoint_state(hp.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters...')
            model.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found, training from scratch...')
            sess.run(tf.global_variables_initializer())

        batches = get_batches(context, keywords, hp.batch_size)
        for e in range(hp.epochs):
            print("----- Epoch {}/{} -----".format(e + 1, hp.epochs))
            steps = 0
            # Keep track of the minimum loss to save best model
            best_loss = 100000.0
            for nextBatch in batches:
                loss = model.train(nextBatch)
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                steps = steps + 1
                if steps % hp.print_loss_steps == 0:
                    print("----- Loss %.2f ----- Perplexity %.2f" % (loss, perplexity))
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                # Only save the best model
                if loss < best_loss and steps % hp.steps_per_checkpoint == 0:
                    best_loss = loss
                    model.saver.save(
                        sess, hp.model_dir + 'seq2seq_epoch{}_step{}_loss{:.2f}.ckpt'.format(e, steps, loss)
                    )
            print()
