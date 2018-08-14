# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import jieba

padToken, unknownToken, goToken, eosToken = 0, 1, 2, 3
from MTA_LSTM.HyperParameter import HyperParameter
import numpy as np


class Batch:
    def __init__(self):
        self.inputs = []
        # self.inputs_length = []
        self.targets = []
        # self.targets_length = []
        self.masks = []
        self.keywords = []


def load_and_cut_data(filepath):

    context = []
    keywords = []
    with open(filepath, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('</d> ')

            data = line[0].replace(' ', '')
            keyword = line[1].replace('\n', '').split(' ')

            seg_list = jieba.cut(data.strip(), cut_all=False)
            cutted_line = [e for e in seg_list]

            context.append(cutted_line)
            keywords.append(keyword)
    return context, keywords


def create_dic_and_map(context, keywords):

    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    # Load dictionary from file
    hp = HyperParameter()
    with open(hp.dictionary_txt, 'r', encoding='utf-8') as f:
        word_dic_new = f.read().split('\n')

    # 将字典中的汉字/英文单词映射为数字
    id_to_word = {idx: word for idx, word in enumerate(special_words + word_dic_new)}
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    # 将sources和targets中的汉字/英文单词映射为数字
    context = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in context]
    keywords = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in keywords]

    return context, keywords, word_to_id, id_to_word


def create_batch(context_, keywords_, num_steps):
    batch = Batch()

    data = np.zeros((len(context_), num_steps+1), dtype=np.int64)

    for i in range(len(context_)):
        temp_data = []
        doc = context_[i]
        temp_data.append([goToken] + doc + [eosToken])
        temp_data = np.array(temp_data, dtype=np.int64)
        data[i][:temp_data.shape[1]] = temp_data

    sources = data[:, 0:num_steps]
    targets = data[:, 1:]
    masks = np.float32(sources != 0)

    batch.inputs = sources
    batch.targets = targets
    batch.masks = masks
    batch.keywords = keywords_

    # batch.inputs_length = [len(source) for source in sources]
    # # len(target) + 1 because of one <EOS>
    # batch.targets_length = [len(target_) + 1 for target_ in targets]
    #
    # max_source_length = max(batch.inputs_length)
    # max_target_length = max(batch.targets_length)

    # for source in sources:
    #     # 将source进行反序并PAD
    #     pad = [padToken] * (max_source_length - len(source))
    #     batch.inputs.append(source + pad)
    #
    # for target_ in targets:
    #     # 将target内的词循环左移
    #     target_new = target_
    #     target_new[:len(target_) - 1] = target_[1:]
    #     target_new[len(target_) - 1] = target_[0]
    #
    #     # 将target进行PAD，并添加EOS符号
    #     pad = [padToken] * (max_target_length - (len(target_new) + 1))
    #     eos = [eosToken] * 1
    #     batch.targets.append(target_new + eos + pad)

    return batch


def get_batches(context, keywords, batch_size):
    data_len = len(context)

    def gen_next_samples():
        for i in range(0, data_len, batch_size):
            yield context[i:min(i + batch_size, data_len)], keywords[i:min(i + batch_size, data_len)]

    batches = []
    for context_, keywords_ in gen_next_samples():
        batch = create_batch(context_, keywords_, 104)
        batches.append(batch)

    return batches


def sentence2enco(sentence, word2id):
    """
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，先将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :return: 处理之后的数据，可直接feed进模型进行预测
    """
    if sentence == '':
        return None
    # 分词
    seg_list = jieba.cut(sentence.strip(), cut_all=False)
    cutted_line = [e for e in seg_list]

    # 将每个单词转化为id
    wordIds = []
    for word in cutted_line:
        wordIds.append(word2id.get(word, unknownToken))
    # 调用createBatch构造batch
    batch = create_batch([wordIds], [[]], [[]])
    return batch


if __name__ == '__main__':

    data_txt = 'data/composition.txt'
    batch_size = 128

    # 得到分词后的sources和targets
    context, keywords = load_and_cut_data(data_txt)

    # 根据sources和targets创建词典，并映射
    context, keywords, word_to_id, id_to_word = create_dic_and_map(context, keywords)
    batches = get_batches(context, keywords, batch_size)

    temp = 0
    for nexBatch in batches:
        if temp == 0:
            print(len(nexBatch.inputs))
            print(nexBatch.inputs)
            print(len(nexBatch.targets))
            print(nexBatch.targets)
            print(len(nexBatch.masks))
            print(nexBatch.masks)
            print(len(nexBatch.keywords))
            print(nexBatch.keywords)
        temp += 1