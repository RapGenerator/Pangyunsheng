import os
from collections import Counter
import configparser as cp

import numpy as np
import jieba


def config_reader():
    """
    Parse model's parameter from configuration_file
    """
    config_dict = {}
    conf = cp.ConfigParser()
    conf.read('./datas/config.ini')

    config_dict['data_folder'] = str(conf.get('DATA', 'DATA_FOLDER'))
    config_dict['batch_size'] = int(conf.get('DATA', 'BATCH_SIZE'))
    config_dict['vocab_size'] = int(conf.get('DATA', 'VOCAB_SIZE'))
    config_dict['window_size'] = int(conf.get('DATA', 'WINDOW_SIZE'))

    config_dict['embedding_dim'] = int(conf.get('MODEL', 'EMBEDDING_DIM'))
    config_dict['num_layers'] = int(conf.get('MODEL', 'NUM_LAYERS'))
    config_dict['num_utils'] = int(conf.get('MODEL', 'NUM_UTILS'))
    config_dict['keep_prob'] = float(conf.get('MODEL', 'KEEP_PROB'))
    config_dict['rnn_mode'] = str(conf.get('MODEL', 'RNN_MODE'))
    config_dict['max_epoch'] = int(conf.get('MODEL', 'MAX_EPOCH'))
    config_dict['learning_rate'] = float(conf.get('MODEL', 'LEARNING_RATE'))

    config_dict['model_path'] = str(conf.get('MODEL', 'MODEL_PATH'))
    config_dict['rhyme_path'] = str(conf.get('MODEL', 'RHYME_PATH'))

    conf.clear()

    return config_dict


class Data(object):
    def __init__(self, config=config_reader()):
        self.data_folder = config['data_folder']
        self.batch_size = config['batch_size']
        self.vocab_size = config['vocab_size']
        self.window_size = config['window_size']

        self.sentences = self._get_sentence()
        self.chunk_size = len(self.sentences) // self.batch_size
        # self.vocab, self.word_to_int, self.int_to_word, self.word_to_count = self._get_vocab()
        self.vocab, self.word_to_int, self.int_to_word = self._get_vocab()

    @staticmethod
    def _get_data(file_path):
        with open(file_path, encoding='utf-8') as f:
            data = f.read()
        return data

    @staticmethod
    def _process_words(data):
        """
        Removal of special symbols
        """
        vocab = sorted(set(data))
        mask = vocab[:4]
        mark = ['\n']
        for m in mask:
            data = data.replace(m, '\\') if m in mark else data.replace(m, '')
        return data

    def _get_sentence(self):
        """
        Read corpus
        """
        data_path = os.path.join(os.getcwd(), self.data_folder, 'raw_data.txt')
        data = self._get_data(data_path)

        # Removal of special symbols
        process_data = self._process_words(data)
        sentence_list = [p for p in process_data.split('\\') if len(p)]
        out_path = os.path.join(os.getcwd(), self.data_folder, 'processed_data.txt')

        # Corpus save and serialization
        with open(out_path, 'w') as outfile:
            for line in sentence_list:
                outfile.write(line+'\n')
        return sentence_list

    def _get_vocab(self):
        # <UNK> represents the word that are not in the dictionary
        # special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
        # sentences = ''.join(self.sentences)
        # # sentences segmentation
        # split_words = list(jieba.cut(sentences, cut_all=False))
        # # The top most_common words
        # word_count = Counter(split_words)
        # word_count = word_count.most_common(self.vocab_size)
        # word_count = {w: c for w, c in word_count}
        # split_words = word_count.keys()
        # vocab = sorted(set(split_words)) + special_words
        with open('datas/dictionary.txt', encoding='utf-8') as f:
            vocab = f.read().split('\n')

        word_to_int = {w: i for i, w in enumerate(vocab)}
        int_to_word = {i: w for i, w in enumerate(vocab)}

        return vocab, word_to_int, int_to_word# , word_count

    def create_dictionary(self):
        special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

        sentences = ''.join(self.sentences)
        split_words = list(jieba.cut(sentences, cut_all=False))
        word_count = Counter(split_words)
        word_count = word_count.most_common(self.vocab_size)
        word_count = {w: c for w, c in word_count}
        split_words = word_count.keys()

        vocab = sorted(set(split_words)) + special_words

        # Save to file
        with open('datas/dictionary.txt', 'w', encoding='utf-8') as f:
            # for word in vocab:
            f.write('\n'.join(vocab))

    def _get_target(self, sentences, index):
        # Get context
        start = index - self.window_size if (index - self.window_size) > 0 else 0
        end = index + 2 * self.window_size
        targets = set(sentences[start:index] + sentences[index + 1:end])
        if len(targets) < 2:
            return False
        else:
            return sentences[start: index], sentences[index + 1: end]

    def get_vector(self, batch):
        if batch != '<EOS>' and batch != '<GO>':
            batch = list(jieba.cut(batch, cut_all=False))[::-1]  # Reverse order
            return [self.word_to_int.get(word, self.word_to_int['<UNK>']) for word in batch]
        else:
            return [self.word_to_int.get(batch, self.word_to_int['<UNK>'])]

    @staticmethod
    def get_batch_length(batch):
        return [len(i) for i in batch]

    def to_full_batch(self, batch):
        max_length = max(self.get_batch_length(batch))
        batch_size = len(batch)
        full_batch = np.full((batch_size, max_length), self.word_to_int['<PAD>'], np.int32)
        for row in range(batch_size):
            full_batch[row, :len(batch[row])] = batch[row]
        return full_batch

    def batch(self):
        # Load corpus
        sentences = self.sentences
        start, end = 0, self.batch_size
        for _ in range(self.chunk_size):
            # batch_x: Central sentence
            # batch_y: The previous sentence
            # batch_z: The next sentence
            # generated size: batch_size - 2
            batch_x, batch_y, batch_z = [], [], []
            for index in range(start, end):
                x = sentences[index]
                targets = self._get_target(sentences, index)
                if targets is False:
                    continue
                y, z = targets
                batch_x.append(x)
                batch_y.append(y[0])
                batch_z.append(z[0])

            encode_x = [self.get_vector(x) for x in batch_x]
            encode_length = self.get_batch_length(encode_x)
            encode_x = self.to_full_batch(encode_x)

            decode_pre_x = [self.get_vector('<GO>') + self.get_vector(y) for y in batch_y]
            decode_pre_y = [self.get_vector(y) + self.get_vector('<EOS>') for y in batch_y]
            decode_pre_length = self.get_batch_length(decode_pre_x)
            decode_pre_x = self.to_full_batch(decode_pre_x)
            decode_pre_y = self.to_full_batch(decode_pre_y)

            decode_post_x = [self.get_vector('<GO>') + self.get_vector(z) for z in batch_z]
            decode_post_y = [self.get_vector(z) + self.get_vector('<EOS>') for z in batch_z]
            decode_post_length = self.get_batch_length(decode_post_x)
            decode_post_x = self.to_full_batch(decode_post_x)
            decode_post_y = self.to_full_batch(decode_post_y)

            yield encode_x, \
                  decode_pre_x, decode_pre_y, \
                  decode_post_x, decode_post_y, \
                  encode_length, decode_pre_length, decode_post_length

            start += self.batch_size
            end += self.batch_size


if __name__ == '__main__':
    data = Data()
    print(len(data.vocab))
    # print(data.vocab)
    # data.create_dictionary()
    # sentences = data.sentences
    # batches = data.batch()
    # for e_x, d_pre_x, d_pre_y, d_post_x, d_post_y, e_length, d_pre_length, d_post_length in batches:
    #     print(d_pre_y[0])