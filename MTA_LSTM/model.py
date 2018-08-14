# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell


class MTALSTM():
    def __init__(self, sess, num_layers, num_steps, rnn_size, embedding_size, word_to_id, num_keywords, learning_rate,
                 max_gradient_norm, is_training, max_to_keep):

        self.sess = sess
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.rnn_size = rnn_size
        self.word_to_id = word_to_id
        self.vocab_size = len(word_to_id)
        self.num_keywords = num_keywords
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.is_training = is_training

        self.batch_size = None
        self.keep_prob = None
        self.inputs = None
        self.targets = None
        self.masks = None
        self.keywords = None
        self.init_output = None
        self.embedding = None
        self.initial_state = None
        self.attention_vs = None
        self._train_op = None

        self.build_graph()
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    def build_graph(self):
        print('Building graph...')

        self.build_placeholder()
        self.build_model()

    def build_placeholder(self):
        print('Building placeholder...')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.masks = tf.placeholder(tf.float32, [None, None], name='masks')
        self.keywords = tf.placeholder(tf.int32, [None, None], name='input_words')
        self.init_output = tf.placeholder(tf.float32, [None, None], name='init_output')

        # embedding
        self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])

    def build_model(self):
        print('Building model...')

        # 创建一个标准的cell
        cell = self.create_rnn_cell()
        # 用cell的zero_state初始化标准cell的initial_state
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        # 把inputs映射成词向量的形式
        inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.inputs)
        # 把keywords映射成词向量的形式
        keywords_embedded = tf.nn.embedding_lookup(self.embedding, self.keywords)

        # 定义coverage变量Ctj
        gate = tf.ones([self.batch_size, self.num_keywords])
        #
        atten_sum = tf.zeros([self.batch_size, self.num_keywords])

        # coverage
        with tf.variable_scope('coverage'):
            # 定义参数矩阵Uf
            u_f = tf.get_variable('u_f', [self.num_keywords * self.embedding_size, self.num_keywords])
            res1 = tf.sigmoid(tf.matmul(tf.reshape(keywords_embedded, [self.batch_size, -1]), u_f))
            # 计算得到Ctj的学习率（前面的东西是什么以及为什么要乘它，没懂）
            phi_res = tf.reduce_sum(self.masks, 1, keepdims=True) * res1

        # LSTM序列
        outputs = []
        output_state = self.init_output
        state = self.initial_state
        with tf.variable_scope('LSTM'):
            for time_step in range(self.num_steps):
                vs = []
                for s2 in range(self.num_keywords):
                    with tf.variable_scope('RNN_attention'):
                        if time_step > 0 or s2 > 0:
                            tf.get_variable_scope().reuse_variables()

                        # 定义attention的参数矩阵和向量
                        u = tf.get_variable('u', [self.rnn_size, 1])
                        w1 = tf.get_variable('w1', [self.rnn_size, self.rnn_size])
                        w2 = tf.get_variable('w2', [self.embedding_size, self.rnn_size])
                        b = tf.get_variable('b', [self.rnn_size])

                        # 得到标准的attention标量值
                        tem1 = tf.matmul(output_state, w1)
                        tem2 = tf.matmul(keywords_embedded[:, s2, :], w2)
                        vi = tf.matmul(tf.tanh(tf.add(tf.add(tem1, tem2), b)), u)
                        # 得到与Ct-1,j相乘后的新attention标量值gtj,append后，得到每个keywords的attention
                        vs.append(vi * gate[:, s2:s2+1])

                # 把每个keywords的attention值concat起来，组成一个[batch_size, num_keywords]的矩阵
                self.attention_vs = tf.concat(vs, axis=1)
                # 对每个gtj进行softmax,得到其0-1的概率值，即atj
                prob_p = tf.nn.softmax(self.attention_vs)

                # 得到atj后，可以根据它的值更新Ctj
                gate = gate - (prob_p / phi_res)

                # (一直没动atten_sum是什么意思)
                atten_sum += prob_p * self.masks[:, time_step:time_step+1]

                # (是不是相当于标准attention的atj*h的步骤，但是为什么h是keywords_embedded)
                mt = tf.add_n([prob_p[:, i:i + 1] * keywords_embedded[:, i, :] for i in range(self.num_keywords)])

                with tf.variable_scope('RNN_sentence'):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()

                    # 数据进入一个cell，得到output和state,
                    (cell_output, state) = cell(tf.concat([inputs_embedded[:, time_step, :], mt], axis=1), state)
                    outputs.append(cell_output)
                    # 需要用来做attention
                    output_state = cell_output

        output = tf.reshape(tf.concat(outputs, axis=1), [-1, self.rnn_size])

        softmax_w = tf.get_variable("softmax_w", [self.rnn_size, self.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.reshape(self.masks, [-1])], average_across_timesteps=False
        )

        cost1 = tf.reduce_sum(loss)
        batch_size = tf.cast(self.batch_size, tf.float32)
        # （这个loss如何理解？）
        cost2 = tf.reduce_sum((phi_res - atten_sum) ** 2)

        self.cost = (cost1 + 0.1 * cost2) / batch_size
        self.prob = tf.nn.softmax(logits)

        if not self.is_training:
            self.sample = tf.arg_max(self.prob, 1)
            return

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def create_rnn_cell(self):

        def single_rnn_cell():
            single_cell = LSTMCell(self.rnn_size)
            basic_cell = DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
            return basic_cell

        cell = MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def train(self, batch):
        initial_output = np.zeros([len(batch.inputs), self.rnn_size])

        feed_dict = {
            self.inputs: batch.inputs,
            self.targets: batch.targets,
            self.masks: batch.masks,
            self.keywords: batch.keywords,
            self.batch_size: len(batch.inputs),
            self.keep_prob: 0.5,
            self.init_output: initial_output
        }

        # state = self.initial_state.eval()
        loss, _ = self.sess.run([self.cost, self._train_op], feed_dict=feed_dict)
        return loss
