# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from tensorflow.contrib.seq2seq import ScheduledEmbeddingTrainingHelper, TrainingHelper, GreedyEmbeddingHelper
from tensorflow.contrib.seq2seq import BasicDecoder, dynamic_decode
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.contrib.seq2seq import tile_batch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class SkipThoughtModel(object):
    def __init__(self, sess, rnn_size, num_layers, embedding_size, word_to_id, mode, use_attention, writer,
                 learning_rate=0.01, max_to_keep=5, beam_search=False, beam_size=5, cell_type='LSTM', max_gradient_norm=5,
                 teacher_forcing=False, teacher_forcing_probability=0.5):

        self.sess = sess
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.word_to_id = word_to_id
        self.vocab_size = len(self.word_to_id)
        self.mode = mode
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.cell_type = cell_type
        self.max_gradient_norm = max_gradient_norm
        self.teacher_forcing = teacher_forcing
        self.teacher_forcing_probability = teacher_forcing_probability
        self.batch_size = None
        self.keep_prob = None
        self.writer = writer

        self.train_op = None
        self.loss_pre = None
        self.loss_post = None

        self.build_graph()
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    def build_graph(self):
        print('Building model...')

        # placeholder
        self.build_placeholder()
        # encoder
        self.build_encoder()
        # decoder
        self.build_decoder()

    def build_placeholder(self):
        print('Building placeholder...')

        # placeholder
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        self.decoder_targets_pre = tf.placeholder(tf.int32, [None, None], name='decoder_targets_pre')
        self.decoder_targets_pre_length = tf.placeholder(tf.int32, [None], name='decoder_targets_pre_length')
        self.decoder_targets_post = tf.placeholder(tf.int32, [None, None], name='decoder_targets_post')
        self.decoder_targets_post_length = tf.placeholder(tf.int32, [None], name='decoder_targets_post_length')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.max_target_pre_sequence_length = tf.reduce_max(self.decoder_targets_pre_length, name='max_target_pre_len')
        self.max_target_post_sequence_length = tf.reduce_max(self.decoder_targets_post_length, name='max_target_post_len')
        self.mask_pre = tf.sequence_mask(
            self.decoder_targets_pre_length,
            self.max_target_pre_sequence_length,
            dtype=tf.float32,
            name='mask_pre'
        )
        self.mask_post = tf.sequence_mask(
            self.decoder_targets_post_length,
            self.max_target_post_sequence_length,
            dtype=tf.float32,
            name='mask_post'
        )

        # embedding矩阵
        self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])

    def build_encoder(self):
        print('Building encoder...')

        with tf.variable_scope('encoder'):
            encoder_cell = self.create_rnn_cell()
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                encoder_cell,
                encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                dtype=tf.float32
            )

    def build_decoder(self):
        print('Building decoder...')

        with tf.variable_scope('decoder'):
            if self.mode == 'train':
                self.build_train_decoder_pre()
                self.build_train_decoder_post()
                self.build_optimizer(self.loss_pre + self.loss_post)
            elif self.mode == 'predict':
                self.build_predict_decoder_pre()
                self.build_predict_decoder_post()
            else:
                raise RuntimeError

    def build_train_decoder_pre(self):
        print('Building train decoder_pre...')

        self.loss_pre = self.build_train_decoder(
            self.decoder_targets_pre,
            self.decoder_targets_pre_length,
            self.max_target_pre_sequence_length,
            self.mask_pre,
            name='pre'
        )

    def build_train_decoder_post(self):
        print('Building train decoder_post...')

        self.loss_post = self.build_train_decoder(
            self.decoder_targets_post,
            self.decoder_targets_post_length,
            self.max_target_post_sequence_length,
            self.mask_post,
            name='post'
        )

    def build_predict_decoder_pre(self):
        print('Building predict decoder_pre...')

        self.decoder_predict_decode_pre = self.build_predict_decoder()

    def build_predict_decoder_post(self):
        print('Building predict decoder_post...')

        self.decoder_predict_decode_post = self.build_predict_decoder()

    def build_train_decoder(self, decoder_targets, decoder_targets_length, max_target_sequence_length, mask, name):
        ending = tf.strided_slice(decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_id['<GO>']), ending], 1)
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, decoder_input)

        decoder_cell, deocder_initial_state = self.build_decoder_cell()
        output_layer = tf.layers.Dense(
            self.vocab_size,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
        )

        if self.teacher_forcing:
            training_helper = ScheduledEmbeddingTrainingHelper(
                inputs=decoder_inputs_embedded,
                sequence_length=decoder_targets_length,
                embedding=self.embedding,
                sampling_probability=self.teacher_forcing_probility,
                time_major=False,
                name='teacher_forcing_training_helper_' + name
            )
        else:
            training_helper = TrainingHelper(
                inputs=decoder_inputs_embedded,
                sequence_length=decoder_targets_length,
                time_major=False,
                name='training_helper_' + name
            )

        training_decoder = BasicDecoder(
            cell=decoder_cell,
            helper=training_helper,
            initial_state=deocder_initial_state,
            output_layer=output_layer
        )

        decoder_outputs, _, _ = dynamic_decode(
            decoder=training_decoder,
            impute_finished=True,
            maximum_iterations=max_target_sequence_length
        )

        decoder_logits_train = tf.identity(decoder_outputs.rnn_output)

        # loss
        loss = sequence_loss(
            logits=decoder_logits_train,
            targets=decoder_targets,
            weights=mask
        )

        return loss

    def build_predict_decoder(self):
        start_tokens = tf.ones([self.batch_size, ], tf.float32) * self.word_to_id['<GO>']
        end_token = self.word_to_id['<EOS>']

        decoder_cell, deocder_initial_state = self.build_decoder_cell()
        output_layer = tf.layers.Dense(
            self.vocab_size,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
        )

        if self.beam_search:
            inference_decoder = BeamSearchDecoder(
                cell=decoder_cell,
                embedding=self.embedding,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=deocder_initial_state,
                beam_width=self.beam_size,
                output_layer=output_layer
            )

        else:
            decoding_helper = GreedyEmbeddingHelper(
                embedding=self.embedding,
                start_tokens=start_tokens,
                end_token=end_token
            )
            inference_decoder = BasicDecoder(
                cell=decoder_cell,
                helper=decoding_helper,
                initial_state=deocder_initial_state,
                output_layer=output_layer
            )

        decoder_outputs, _, _ = dynamic_decode(decoder=inference_decoder, maximum_iterations=50)

        if self.beam_search:
            decoder_predict_decode = decoder_outputs.predicted_ids
        else:
            decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)

        return decoder_predict_decode

    def build_decoder_cell(self):
        encoder_inputs_length = self.encoder_inputs_length
        if self.beam_search:
            print('Use beamsearch decoding...')
            self.encoder_outputs = tile_batch(self.encoder_outputs, multiplier=self.beam_size)
            self.encoder_state = nest.map_structure(lambda s: tile_batch(s, self.beam_size), self.encoder_state)
            encoder_inputs_length = tile_batch(encoder_inputs_length, multiplier=self.beam_size)

        # 定义要使用的attention机制
        attention_mechanism = BahdanauAttention(
            num_units=self.rnn_size,
            memory=self.encoder_outputs,
            memory_sequence_length=encoder_inputs_length
        )

        # 定义decoder阶段要用的RNNCell，然后为其封装attention wrapper
        decoder_cell = self.create_rnn_cell()
        decoder_cell = AttentionWrapper(
            cell=decoder_cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.rnn_size,
            name='Attention_Wrapper'
        )

        batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

        decoder_initial_state = decoder_cell.zero_state(
            batch_size=batch_size,
            dtype=tf.float32).clone(
            cell_state=self.encoder_state
        )

        return decoder_cell, decoder_initial_state

    def create_rnn_cell(self):
        def single_rnn_cell():
            single_cell = GRUCell(self.rnn_size) if self.cell_type == 'GRU' else LSTMCell(self.rnn_size)
            basic_cell = DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
            return basic_cell
        cell = MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_optimizer(self, loss):
        print('Building optimizer...')

        # optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

    def train(self, batch):
        feed_dict = {
            self.encoder_inputs: batch.encoder_inputs,
            self.encoder_inputs_length: batch.encoder_inputs_length,
            self.decoder_targets_pre: batch.decoder_targets_pre,
            self.decoder_targets_pre_length: batch.decoder_targets_pre_length,
            self.decoder_targets_post: batch.decoder_targets_post,
            self.decoder_targets_post_length: batch.decoder_targets_post_length,
            self.batch_size: len(batch.encoder_inputs),
            self.keep_prob: 0.5
        }
        _, loss_pre, loss_post = self.sess.run([self.train_op, self.loss_pre, self.loss_post], feed_dict=feed_dict)
        return loss_pre, loss_post

    def eval(self, batch):
        feed_dict = {
            self.encoder_inputs: batch.encoder_inputs,
            self.encoder_inputs_length: batch.encoder_inputs_length,
            self.decoder_targets_pre: batch.decoder_targets_pre,
            self.decoder_targets_pre_length: batch.decoder_targets_pre_length,
            self.decoder_targets_post: batch.decoder_targets_post,
            self.decoder_targets_post_length: batch.decoder_targets_post_length,
            self.batch_size: len(batch.encoder_inputs),
            self.keep_prob: 1.0
        }
        loss_pre, loss_post = self.sess.run([self.loss_pre, self.loss_post], feed_dict=feed_dict)
        return loss_pre, loss_post

    def infer(self, batch):
        feed_dict = {
            self.encoder_inputs: batch.encoder_inputs,
            self.encoder_inputs_length: batch.encoder_inputs_length,
            self.batch_size: len(batch.encoder_inputs),
            self.keep_prob: 1.0
        }
        predict_pre, predict_post = self.sess.run([self.decoder_predict_decode_pre, self.build_predict_decoder_post()],
                                                  feed_dict=feed_dict)
        return predict_pre, predict_post