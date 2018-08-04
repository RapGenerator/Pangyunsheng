# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf


class Seq2SeqModel(object):
    def __init__(self, rnn_size, num_layers, embedding_size, learning_rate, word_to_idx, mode, use_attention,
                 beam_search, beam_size, max_gradient_norm=5.0):
        self.learing_rate = learning_rate
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.word_to_idx = word_to_idx
        self.vocab_size = len(self.word_to_idx)
        self.mode = mode
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.max_gradient_norm = max_gradient_norm
        self.__graph__()
        self.saver = tf.train.Saver()

    def __graph__(self):

        # placeholder
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32,
                                     name='masks')

        # embedding矩阵,encoder和decoder共用该词向量矩阵
        self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])

        # encoder
        encoder_outputs, encoder_state = self.encoder()

        # decoder
        with tf.variable_scope('decoder'):
            # 定义要使用的attention机制。
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.rnn_size, memory=encoder_outputs,
                memory_sequence_length=self.encoder_inputs_length
            )
            # 定义decoder阶段要是用的LSTMCell，然后为其封装attention wrapper
            decoder_cell = self.create_rnn_cell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.rnn_size,
                name='Attention_Wrapper'
            )

            # 定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
            decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(
                cell_state=encoder_state)
            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(
                mean=0.0, stddev=0.1))

            if self.mode == 'train':
                self.decoder_outputs = self.decoder_train(decoder_cell, decoder_initial_state, output_layer)
                # loss
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_outputs, targets=self.decoder_targets,
                                                             weights=self.mask)

                # summary
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()

                # optimizer
                optimizer = tf.train.AdamOptimizer(self.learing_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
            elif self.mode == 'decode':
                self.decoder_predict_decode = self.decoder_decode(decoder_cell, decoder_initial_state, output_layer)

    def encoder(self):
        """
        创建模型的encoder部分
        :return: encoder_outputs: 用于attention，batch_size*encoder_inputs_length*rnn_size
                 encoder_state: 用于decoder的初始化状态，batch_size*rnn_size
        """
        with tf.variable_scope('encoder'):
            # 创建LSTMCell，两层+dropout
            encoder_cell = self.create_rnn_cell()
            # encoder_inputs的词向量
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            # 使用dynamic_rnn构建LSTM模型，将输入编码成隐层向量。
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                               sequence_length=self.encoder_inputs_length,
                                                               dtype=tf.float32)
            return encoder_outputs, encoder_state

    def decoder_train(self, decoder_cell, decoder_initial_state, output_layer):
        """
        创建train的decoder部分
        :param encoder_outputs: encoder的输出
        :param encoder_state: encoder的state
        :return: decoder_logits_train: decoder的predict
        """
        # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<end>,并进行embedding。
        # decoder_inputs_embedded的shape为[batch_size, decoder_targets_length, embedding_size]
        ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<GO>']), ending], 1)
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, decoder_input)
        # 训练阶段，使用TrainingHelper+BasicDecoder的组合
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                            sequence_length=self.decoder_targets_length,
                                                            time_major=False, name='training_helper')
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                           initial_state=decoder_initial_state,
                                                           output_layer=output_layer)
        # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
        # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
        # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                  impute_finished=True,
                                                                  maximum_iterations=self.max_target_sequence_length)
        # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
        decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
        # decoder_predict_train = tf.argmax(decoder_logits_train, axis=-1, name='decoder_pred_train')

        return decoder_logits_train

    def decoder_decode(self, decoder_cell, decoder_initial_state, output_layer):
        start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<GO>']
        end_token = self.word_to_idx['<EOS>']

        decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding, start_tokens=start_tokens,
                                                                   end_token=end_token)
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                            initial_state=decoder_initial_state,
                                                            output_layer=output_layer)
        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, maximum_iterations=8)
        decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)

        return decoder_predict_decode

    def create_rnn_cell(self):
        """
        创建标准的RNN Cell，相当于一个时刻的Cell
        :return: cell: 一个Deep RNN Cell
        """

        def single_rnn_cell():
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            # dropout
            basic_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
            return basic_cell

        # 列表中每个元素都是调用single_rnn_cell函数
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def train(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob: 0.5,
                     self.batch_size: len(batch.encoder_inputs)}
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def eval(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict
