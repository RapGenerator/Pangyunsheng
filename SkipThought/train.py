# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from SkipThought.DataHelpers import *
from SkipThought.model import SkipThoughtModel
from SkipThought.HyperParameter import HyperParameter
import math
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
    print_loss_steps = hp.print_loss_steps
    beam_size = hp.beam_size
    teacher_forcing = hp.teacher_forcing
    teacher_forcing_probability = hp.teacher_forcing_probability
    max_to_keep = hp.max_to_keep
    writer = None

    # 得到分词后的data
    data = load_and_cut_data(data_txt)

    data, word_to_id, _ = create_dic_and_map(data)

    # Train
    with tf.Session() as sess:
        # Build model
        # Note that beam_search should be False while training!!!
        model = SkipThoughtModel(
            sess=sess,
            rnn_size=rnn_size,
            num_layers=num_layers,
            embedding_size=embedding_size,
            word_to_id=word_to_id,
            mode='train',
            learning_rate=learning_rate,
            use_attention=True,
            beam_search=False,
            beam_size=beam_size,
            teacher_forcing=teacher_forcing,
            teacher_forcing_probability=teacher_forcing_probability,
            cell_type='LSTM',
            max_gradient_norm=5.0,
            max_to_keep=max_to_keep,
            writer=writer
        )

        # Trying to restore model
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters...')
            model.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found, training from scratch...')
            sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            print("----- Epoch {}/{} -----".format(e + 1, epochs))
            batches = get_batches(data, batch_size)
            steps = 0
            # Keep track of the minimum loss to save best model
            best_loss = 100000.0
            for nextBatch in batches:
                loss_pre, loss_post = model.train(nextBatch)
                perplexity = math.exp(float(loss_pre + loss_post)) if (loss_pre + loss_post) < 300 else float('inf')
                steps = steps + 1
                if steps % print_loss_steps == 0:
                    print("----- Loss_pre %.2f -----Loss_post %.2f----- Perplexity %.2f" % (loss_pre, loss_post, perplexity))
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                # Only save the best model
                if (loss_pre + loss_post) < best_loss and steps % steps_per_checkpoint == 0:
                    best_loss = (loss_pre + loss_post)
                    model.saver.save(
                        sess, model_dir + 'seq2seq_epoch{}_step{}_loss{:.2f}.ckpt'.format(e, steps, (loss_pre + loss_post))
                    )
            print()
