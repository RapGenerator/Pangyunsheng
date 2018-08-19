import jieba
import tensorflow as tf
import numpy as np
from model import Model
from data_utils import config_reader
from generate_utils import Beam, sort_prob, RhymeChecker, get_next_sentence_rhyme_word, sort_word_by_prob, \
    get_sort_word_by_prob


class Gen(object):
    def __init__(self):
        # Load model config
        self.config = config_reader()
        self.model_path = self.config['model_path']
        self.rhyme_checker = RhymeChecker(self.config['rhyme_path'])

        # Model initial and rebuild graph
        self.model = Model()
        self.encode, _, _, self.decode_post_x, _, self.encode_length, _, self.decode_post_length = self.model.build_inputs()
        self.encode_emb, _, self.decode_post_emb = self.model.build_word_embedding(self.encode,
                                                                                   self.decode_post_x,
                                                                                   self.decode_post_x)
        self.initial_state, self.final_state = self.model.build_encoder(self.encode_emb,
                                                                        self.encode_length,
                                                                        train=False)
        _, self.post_prediction, self.post_state = self.model.build_decoder(self.decode_post_emb,
                                                                            self.decode_post_length,
                                                                            self.final_state,
                                                                            scope='decoder_post')

        # Initial session and restore model weight
        # saver = tf.train.Saver()
        # self.sess = tf.Session()
        # saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

    def restore_model(self, model_path):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

    def user_input(self, text, rhyme_style, sample_size, target_long=8, rhyme_mode=1, rhyme_change_gap=4, beam_width=20):
        """
        :param text: pre-text, input from users
        :param sample_size: length of sentence generate
        :param rhyme_mode: set the rhyme_mode: 1(default) for last one char rhyme, 2 for last two char rhyme
        :param rhyme_change_gap: Change a rhyme for every 'rhyme_change_gap' sentence
        :param beam_width: [int], width of beam search
        :return:
        """
        self.input_text = text
        self.rhyme_style = rhyme_style
        self.sample_size = sample_size
        self.target_long = target_long
        self.rhyme_mode = rhyme_mode
        self.rhyme_change_gap = rhyme_change_gap
        self.beam_width = beam_width

    def rhyme_style_pai(self, count, predict, last_word, used_words):

        if count % self.rhyme_change_gap != 0:  # 不换韵
            count += 1
            rhyme_words, rhyme_words_index = get_next_sentence_rhyme_word(
                last_word=last_word,
                rhyme_mode=self.rhyme_mode,
                result_rhyme_word_len=2,
                prob=predict[-1],
                int_to_word=self.model.data.int_to_word,
                word_to_int=self.model.data.word_to_int,
                used_words=used_words
            )
            used_words.append(rhyme_words[0])
            first_word = rhyme_words_index[0]
        else:  # 换韵
            count = 1
            first_word = get_sort_word_by_prob(self.model.data.int_to_word, predict[-1])
            used_words.append(first_word)
            last_word = first_word
            first_word = self.model.data.word_to_int[first_word]

        return count, first_word, last_word, used_words

    def rhyme_style_jiao(self, count, predict, last_word_a, last_word_b, used_words):

        if count % 2 == 0:  # A韵
            rhyme_words, rhyme_words_index = get_next_sentence_rhyme_word(
                last_word=last_word_a,
                rhyme_mode=self.rhyme_mode,
                result_rhyme_word_len=2,
                prob=predict[-1],
                int_to_word=self.model.data.int_to_word,
                word_to_int=self.model.data.word_to_int,
                used_words=used_words
            )
            used_words.append(rhyme_words[0])
            first_word = rhyme_words_index[0]
        else:  # B韵
            rhyme_words, rhyme_words_index = get_next_sentence_rhyme_word(
                last_word=last_word_b,
                rhyme_mode=self.rhyme_mode,
                result_rhyme_word_len=2,
                prob=predict[-1],
                int_to_word=self.model.data.int_to_word,
                word_to_int=self.model.data.word_to_int,
                used_words=used_words
            )
            used_words.append(rhyme_words[0])
            first_word = rhyme_words_index[0]
        count += 1

        return count, first_word, used_words

    def rhyme_style_gehang(self, count, predict, last_word_a, used_words):

        if count % 2 == 0:  # 不押韵
            first_word = get_sort_word_by_prob(self.model.data.int_to_word, predict[-1])
            used_words.append(first_word)
            first_word = self.model.data.word_to_int[first_word]
        else:  # 押韵
            rhyme_words, rhyme_words_index = get_next_sentence_rhyme_word(
                last_word=last_word_a,
                rhyme_mode=self.rhyme_mode,
                result_rhyme_word_len=2,
                prob=predict[-1],
                int_to_word=self.model.data.int_to_word,
                word_to_int=self.model.data.word_to_int,
                used_words=used_words
            )
            used_words.append(rhyme_words[0])
            first_word = rhyme_words_index[0]
        count += 1

        return count, first_word, used_words

    def rhyme_style_bao(self, count, predict, last_word_a, last_word_b, used_words):

        if count == 1 or count == 2:  # B韵
            rhyme_words, rhyme_words_index = get_next_sentence_rhyme_word(
                last_word=last_word_b,
                rhyme_mode=self.rhyme_mode,
                result_rhyme_word_len=2,
                prob=predict[-1],
                int_to_word=self.model.data.int_to_word,
                word_to_int=self.model.data.word_to_int,
                used_words=used_words
            )
            used_words.append(rhyme_words[0])
            first_word = rhyme_words_index[0]
        else:  # A韵
            rhyme_words, rhyme_words_index = get_next_sentence_rhyme_word(
                last_word=last_word_a,
                rhyme_mode=self.rhyme_mode,
                result_rhyme_word_len=2,
                prob=predict[-1],
                int_to_word=self.model.data.int_to_word,
                word_to_int=self.model.data.word_to_int,
                used_words=used_words
            )
            used_words.append(rhyme_words[0])
            first_word = rhyme_words_index[0]
        count += 1

        return count, first_word, used_words

    def use_beam_search(self, start_word, encode_x, decode_x, samples, sample_i, last_word):

        # Initial an object of beam search
        beam_searcher = Beam(width=self.beam_width,
                             stop_index=self.model.data.word_to_int['<EOS>'],
                             index2word=self.model.data.int_to_word,
                             start_len=len(last_word),
                             ensure_len=False)

        ignore_words = ['<UNK>']
        ignore_index = [self.model.data.word_to_int[word] for word in ignore_words]

        while not beam_searcher.check_finished():
            beams = beam_searcher.beams
            beam_count = 0

            for beam in beams:
                beam_indexes = [node.index for node in beam]
                decode_x[0] = start_word + beam_indexes[1:]

                if beam[-1].stopped:
                    beam_count += 1
                    continue

                feed = {self.encode: encode_x,
                        self.encode_length: [len(encode_x[0])],
                        # self.initial_state: new_state,
                        self.decode_post_x: decode_x,
                        self.decode_post_length: [len(decode_x[0])]}
                predict, state = self.sess.run([self.post_prediction, self.post_state], feed_dict=feed)

                sorted_probs = sort_prob(predict[-1])
                high_probs = []
                for item in sorted_probs:
                    if item[0] not in ignore_index:
                        high_probs.append(item)
                    if len(high_probs) == self.beam_width:
                        break
                for prob in high_probs:
                    beam_searcher.add_prob(prob[1], prob[0], state, beam_count)
                beam_count += 1

            beam_searcher.shrink_beam()
        best_line_index, best_beam = beam_searcher.get_best()
        decode_x[0] = start_word + best_line_index
        samples[sample_i] += decode_x[0][1:-1]
        encode_x = [samples[sample_i]]
        # new_state = state
        output = ''.join([self.model.data.int_to_word[sample] for sample in samples[sample_i]][::-1])
        return output, encode_x

    def generator(self):
        """
        :return: generate every sentence
        """
        encode_x = [self.model.data.get_vector(self.input_text)]
        decode_x = [[self.model.data.word_to_int['<GO>']]]

        samples = [[] for _ in range(self.sample_size + 1)]
        samples[0] = encode_x[0]

        start_index = [self.model.data.word_to_int['<GO>']]

        last_word = list(jieba.cut(self.input_text, cut_all=False))[-1]
        used_words = []

        def get_feed(encode_x):
            feed = {self.encode: encode_x,
                    self.encode_length: [len(encode_x[0])],
                    self.decode_post_x: decode_x,
                    self.decode_post_length: [len(decode_x[0])]}
            return feed

        if self.rhyme_style == 'AAAA':
            print('Rhyme style: pai')

            count = 1
            for sample_i in range(1, self.sample_size + 1):
                # 预测第一个词的概率分布（句尾押韵词）
                predict, _ = self.sess.run([self.post_prediction, self.post_state], feed_dict=get_feed(encode_x))
                # 得到第一个符合押韵的词（first_word）
                count, first_word, last_word, used_words = self.rhyme_style_pai(
                    count,
                    predict,
                    last_word,
                    used_words
                )
                # 根据第一个词生成后面的词
                start_word = start_index + [first_word]
                sentence, encode_x = self.use_beam_search(start_word, encode_x, decode_x, samples, sample_i, last_word)

                yield sentence

        elif self.rhyme_style == 'ABAB':
            print('Rhyme style: jiao')

            # 先得到下一句第一个词的概率分布
            predict, _ = self.sess.run([self.post_prediction, self.post_state], feed_dict=get_feed(encode_x))

            # 得到A和B的韵（A的韵为输入句，B的韵为输入句的下一句）
            last_word_a = last_word
            last_word_b = get_sort_word_by_prob(self.model.data.int_to_word, predict[-1])

            count = 1
            for sample_i in range(1, self.sample_size + 1):

                predict, state = self.sess.run([self.post_prediction, self.post_state], feed_dict=get_feed(encode_x))
                count, first_word, used_words = self.rhyme_style_jiao(
                    count,
                    predict,
                    last_word_a,
                    last_word_b,
                    used_words
                )

                start_word = start_index + [first_word]
                sentence, encode_x = self.use_beam_search(start_word, encode_x, decode_x, samples, sample_i, last_word)

                yield sentence

        elif self.rhyme_style == '_A_A':
            print('Rhyme style: gehang')

            predict, state = self.sess.run([self.post_prediction, self.post_state], feed_dict=get_feed(encode_x))

            # 得到A的韵
            last_word = get_sort_word_by_prob(self.model.data.int_to_word, predict[-1])

            count = 1
            for sample_i in range(1, self.sample_size + 1):

                predict, state = self.sess.run([self.post_prediction, self.post_state], feed_dict=get_feed(encode_x))
                count, first_word, used_words = self.rhyme_style_gehang(
                    count,
                    predict,
                    last_word,
                    used_words
                )

                start_word = start_index + [first_word]
                sentence, encode_x = self.use_beam_search(start_word, encode_x, decode_x, samples, sample_i, last_word)

                yield sentence

        elif self.rhyme_style == 'ABBA':
            print('Rhyme style: bao')

            # 先得到下一句第一个词的概率分布
            predict, _ = self.sess.run([self.post_prediction, self.post_state], feed_dict=get_feed(encode_x))

            # 得到A和B的韵（A的韵为输入句，B的韵为输入句的下一句）
            last_word_a = last_word
            last_word_b = get_sort_word_by_prob(self.model.data.int_to_word, predict[-1])

            count = 1
            for sample_i in range(1, 4):
                predict, state = self.sess.run([self.post_prediction, self.post_state], feed_dict=get_feed(encode_x))
                count, first_word, used_words = self.rhyme_style_bao(
                    count,
                    predict,
                    last_word_a,
                    last_word_b,
                    used_words
                )

                start_word = start_index + [first_word]
                sentence, encode_x = self.use_beam_search(start_word, encode_x, decode_x, samples, sample_i, last_word)

                yield sentence

        else:
            raise RuntimeError

        self.sess.close()


def get_sentences(gen, model_path, input_text, rhyme_mode, rhyme_style_id, sample_size, target_long):
    """

    :param gen: model
    :param input_text: 由主题词生成的一句文本
    :param rhyme_mode: 单押/双押，1为单押，2为双押
    :param rhyme_style_id: 押韵style的id，0为排韵，1为交韵，2为隔行韵，3为抱韵
    :param sample_size: 要生成的歌词行数
    :param target_long: 每句歌词的字数
    :return: 生成的歌词
    """
    lyrics = []
    rhyme_style = ['AAAA', 'ABAB', '_A_A', 'ABBA']

    gen.restore_model(model_path)
    gen.user_input(
        text=input_text,
        sample_size=sample_size,
        target_long=target_long,
        rhyme_mode=rhyme_mode,
        rhyme_style=rhyme_style[rhyme_style_id]
    )
    sentences = gen.generator()
    for sen in sentences:
        lyrics.append(sen)

    return lyrics


if __name__ == '__main__':
    # rhyme_mode = [1, 2]
    #
    # input_text = '人民广场吃炸鸡'
    # print(input_text)
    # gen = Gen()
    # gen.user_input(text=input_text, rhyme_style=rhyme_style[0])
    # sentences = gen.generator()
    # for sen in sentences:
    #     print(sen)

    gen = Gen()
    model_path = './checkpoint'
    input_text = '腿搁在办公桌上'
    rhyme_mode = 1
    rhyme_style_id = 0
    sample_size = 10
    target_long = 8

    lyrics = get_sentences(gen, model_path, input_text, rhyme_mode, rhyme_style_id, sample_size, target_long)
    for ly in lyrics:
        print(ly)
