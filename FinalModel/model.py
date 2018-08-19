import tensorflow as tf
from tensorflow.contrib import rnn

from data_utils import Data, config_reader


class Model(object):
    def __init__(self, config=config_reader()):
        """
        :Desc: read model param
        :param config:
        """
        self.rnn_mode = config['rnn_mode']
        self.batch_size = config['batch_size'] - 2
        self.embedding_dim = config['embedding_dim']
        self.num_layers = config['num_layers']
        self.num_units = config['num_utils']
        self.learning_rate = config['learning_rate']
        self.max_epoch = config['max_epoch']
        self.keep_prob = config['keep_prob']

        self.data = Data()
        self.vocab = self.data.vocab
        self.chunk_size = self.data.chunk_size
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)

    @staticmethod
    def soft_max_variable(num_units, vocab_size, reuse=False):
        with tf.variable_scope('soft_max', reuse=reuse):
            w = tf.get_variable("w", [num_units, vocab_size])
            b = tf.get_variable("b", [vocab_size])
        return w, b

    @staticmethod
    def build_inputs():
        with tf.variable_scope('inputs'):
            # Central sentence
            encode = tf.placeholder(tf.int32, shape=[None, None], name='encode')
            encode_length = tf.placeholder(tf.int32, shape=[None, ], name='encode_length')

            # pre sentence
            decode_pre_x = tf.placeholder(tf.int32, shape=[None, None], name='decode_pre_x')
            decode_pre_y = tf.placeholder(tf.int32, shape=[None, None], name='decode_pre_y')
            decode_pre_length = tf.placeholder(tf.int32, shape=[None, ], name='decode_pre_length')

            # post sentence
            decode_post_x = tf.placeholder(tf.int32, shape=[None, None], name='decode_post_x')
            decode_post_y = tf.placeholder(tf.int32, shape=[None, None], name='decode_post_y')
            decode_post_length = tf.placeholder(tf.int32, shape=[None, ], name='decode_post_length')

        return encode, decode_pre_x, decode_pre_y,  decode_post_x, decode_post_y, encode_length, decode_pre_length, decode_post_length

    def build_word_embedding(self, encode, decode_pre_x, decode_post_x):
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable(name='embedding', shape=[len(self.vocab), self.embedding_dim],
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1))
            encode_emb = tf.nn.embedding_lookup(embedding, encode, name='encode_emb')
            decode_pre_emb = tf.nn.embedding_lookup(embedding, decode_pre_x, name='decode_pre_emb')
            decode_post_emb = tf.nn.embedding_lookup(embedding, decode_post_x, name='decode_post_emb')
        return encode_emb, decode_pre_emb, decode_post_emb

    def build_encoder(self, encode_emb, length, scope='encoder', train=True):
        batch_size = self.batch_size if train else 1
        if self.rnn_mode == 'Bi-directional':
            with tf.variable_scope(self.rnn_mode+scope):
                # Forward direction
                lstm_fw_cell = rnn.BasicLSTMCell(num_units=self.num_units, forget_bias=1.0)
                # Backward direction
                lstm_bw_cell = rnn.BasicLSTMCell(num_units=self.num_units, forget_bias=1.0)

                initial_state_fw = lstm_fw_cell.zero_state(batch_size, tf.float32)
                initial_state_bw = lstm_bw_cell.zero_state(batch_size, tf.float32)

                initial_state_c = tf.reduce_mean(tf.stack([initial_state_fw.c, initial_state_bw.c], axis=0), axis=0)
                initial_state_h = tf.reduce_mean(tf.stack([initial_state_fw.h, initial_state_bw.h], axis=0), axis=0)
                initial_state = rnn.LSTMStateTuple(initial_state_c, initial_state_h)

                # inputs:[batch_size,n_steps,n_input]
                hiddens, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 initial_state_fw=initial_state_fw,
                                                                 initial_state_bw=initial_state_bw,
                                                                 inputs=encode_emb,
                                                                 sequence_length=length,
                                                                 dtype=tf.float32)

                # state tuple (fw's {LSTMStateTuple},bw's{LSTMStateTuple})
                # LSTMStateTuple:(c=(?, num_units),h=(?, num_units))
                state_c = tf.reduce_mean(tf.stack([state[0].c, state[1].c], axis=0), axis=0)
                state_h = tf.reduce_mean(tf.stack([state[0].h, state[1].h], axis=0), axis=0)
                state_final = rnn.LSTMStateTuple(state_c, state_h)
            return initial_state, state_final

        elif self.rnn_mode == 'multilayer':
            with tf.variable_scope(self.rnn_mode+scope):
                # build two layers LSTM
                stack_rnn = []
                for i in range(self.num_layers):
                    cell = rnn.BasicLSTMCell(num_units=self.num_units)
                    drop_cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=self.keep_prob)
                    stack_rnn.append(drop_cell)
                cell = rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)
                initial_state = cell.zero_state(batch_size, tf.float32)
                _, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                   inputs=encode_emb,
                                                   initial_state=initial_state,
                                                   sequence_length=length)
            return initial_state, final_state

        else:
            pass

    def build_decoder(self, decode_emb, length, state, scope='decoder', reuse=False):
        if self.rnn_mode == 'Bi-directional':
            with tf.variable_scope(self.rnn_mode+scope):
                cell = rnn.BasicLSTMCell(num_units=self.num_units)
                outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                         inputs=decode_emb,
                                                         initial_state=state,
                                                         sequence_length=length)
            x = tf.reshape(outputs, [-1, self.num_units])
            w, b = self.soft_max_variable(self.num_units, len(self.vocab), reuse=reuse)
            logits = tf.matmul(x, w) + b
            prediction = tf.nn.softmax(logits, name='predictions')
            return logits, prediction, final_state

        elif self.rnn_mode == ' multilayer':
            with tf.variable_scope(self.rnn_mode+scope):
                stack_rnn = []
                for i in range(self.num_layers):
                    cell = rnn.BasicLSTMCell(num_units=self.num_units)
                    drop_cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                    stack_rnn.append(drop_cell)
                cell = rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)
                outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                         inputs=decode_emb,
                                                         initial_state=state,
                                                         sequence_length=length)
            x = tf.reshape(outputs, [-1, self.num_units])
            w, b = self.soft_max_variable(self.num_units, len(self.vocab), reuse=reuse)
            logits = tf.matmul(x, w) + b
            prediction = tf.nn.softmax(logits, name='predictions')
            return logits, prediction, final_state

        else:
            pass

    def build_loss(self, logits, targets, scope='loss'):
        with tf.variable_scope(scope):
            y_one_hot = tf.one_hot(targets, len(self.vocab))
            y_reshaped = tf.reshape(y_one_hot, [-1, len(self.vocab)])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
            # res2 = tf.contrib.seq2seq
        return loss

    def build_optimizer(self, loss, scope='optimizer'):
        with tf.variable_scope(scope):
            grad_clip = 5
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
            train_op = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer

    def build(self):
        # input placeholder
        encode, decode_pre_x, decode_pre_y, decode_post_x, decode_post_y, encode_length, decode_pre_length, decode_post_length = self.build_inputs()

        # embedding
        encode_emb, decode_pre_emb, decode_post_emb = self.build_word_embedding(encode, decode_pre_x, decode_post_x)

        # build encoder
        initial_state, final_state = self.build_encoder(encode_emb, encode_length)

        # build pre sentence decoder
        pre_logits, pre_prediction, pre_state = self.build_decoder(decode_pre_emb, decode_pre_length, final_state,
                                                                   scope='decoder_pre')
        pre_loss = self.build_loss(pre_logits, decode_pre_y, scope='decoder_pre_loss')
        pre_optimizer = self.build_optimizer(pre_loss, scope='decoder_pre_op')
        # build post sentence decoder
        post_logits, post_prediction, post_state = self.build_decoder(decode_post_emb, decode_post_length, final_state,
                                                                      scope='decoder_post', reuse=True)
        post_loss = self.build_loss(post_logits, decode_post_y, scope='decoder_post_loss')
        post_optimizer = self.build_optimizer(post_loss, scope='decoder_post_op')

        inputs = {'initial_state': initial_state, 'encode': encode, 'encode_length': encode_length,
                  'decode_pre_x': decode_pre_x, 'decode_pre_y': decode_pre_y, 'decode_pre_length': decode_pre_length,
                  'decode_post_x':  decode_post_x, 'decode_post_y': decode_post_y, 'decode_post_length': decode_post_length}
        decode_pre = {'pre_optimizer': pre_optimizer, 'pre_loss': pre_loss, 'pre_state': pre_state}
        decode_post = {'post_optimizer': post_optimizer, 'post_loss': post_loss, 'post_state': post_state}

        return inputs, decode_pre, decode_post

    # def build_predict(self):
    #     encode, _, _, decode_post_x, _, encode_length, _, decode_post_length = self.build_inputs()
    #
    #     encode_emb, _, decode_post_emb = self.build_word_embedding(encode, decode_post_x, decode_post_x)
    #
    #     initial_state, final_state = self.build_encoder(encode_emb, encode_length, train=False)
    #
    #     _, post_prediction, post_state = self.build_decoder(decode_post_emb, decode_post_length, final_state,
    #                                                         scope='decoder_post')
    #
    #     inputs = {'initial_state': initial_state, 'encode': encode, 'encode_length': encode_length,
    #               'decode_post_x': decode_post_x, 'decode_post_length': decode_post_length}
    #
    #     return inputs, post_prediction, post_state

    @staticmethod
    def restore(sess, saver, path):
        saver.restore(sess, save_path=path)
        print('Model restored from {}'.format(path))