# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import jieba
from SkipThought.HyperParameter import HyperParameter

padToken, unknownToken, goToken, eosToken = 0, 1, 2, 3


class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets_pre = []
        self.decoder_targets_pre_length = []
        self.decoder_targets_post = []
        self.decoder_targets_post_length = []


def load_and_cut_data(filepath):
    """
    加载数据并分词
    :param filepath: 路径
    :return: data: 分词后的数据
    """
    with open(filepath, 'r', encoding='UTF-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            seg_list = jieba.cut(line.strip(), cut_all=False)
            cutted_line = [e for e in seg_list]
            data.append(cutted_line)
    return data


def create_dic_and_map(data_txt):
    """
    得到输入和输出的字符映射表
    :param sources:
           targets:
    :return: sources_data:
             targets_data:
             word_to_id: 字典，数字到数字的转换
             id_to_word: 字典，数字到汉字的转换
    """
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    # Load dictionary from file
    hp = HyperParameter()
    with open(hp.dictionary_txt, 'r', encoding='utf-8') as f:
        word_dic_new = f.read().split('\n')

    # 将字典中的汉字/英文单词映射为数字
    id_to_word = {idx: word for idx, word in enumerate(special_words + word_dic_new)}
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    # 将sources和targets中的汉字/英文单词映射为数字
    data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in data_txt]

    return data, word_to_id, id_to_word


def create_batch(sources, targets_pre, targets_post):
    batch = Batch()
    batch.encoder_inputs_length = [len(source) for source in sources]
    # len(target) + 1 because of one <EOS>
    batch.decoder_targets_pre_length = [len(target_pre) + 1 for target_pre in targets_pre]
    batch.decoder_targets_post_length = [len(target_post) + 1 for target_post in targets_post]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_pre_length = max(batch.decoder_targets_pre_length)
    max_target_post_length = max(batch.decoder_targets_post_length)

    for source in sources:
        # 将source进行反序并PAD
        source = list(reversed(source))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

    for target_pre in targets_pre:
        # 将target进行PAD，并添加EOS符号
        pad = [padToken] * (max_target_pre_length - (len(target_pre) + 1))
        eos = [eosToken] * 1
        batch.decoder_targets_pre.append(target_pre + eos + pad)

    for target_post in targets_post:
        # 将target进行PAD，并添加EOS符号
        pad = [padToken] * (max_target_post_length - (len(target_post) + 1))
        eos = [eosToken] * 1
        batch.decoder_targets_post.append(target_post + eos + pad)

    return batch


def get_batches(data, batch_size):
    data_len = len(data)

    def gen_next_samples():
        for i in range(data_len //batch_size + 1):
            yield data[(i*batch_size+1):min((i*batch_size+1) + batch_size, data_len-1)], \
                  data[(i*batch_size):min((i*batch_size) + batch_size, data_len-2)], \
                  data[(i*batch_size+2):min((i*batch_size+2) + batch_size, data_len)]

    batches = []
    for sources, targets_pre, targets_post in gen_next_samples():
        batch = create_batch(sources, targets_pre, targets_post)
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
    batch = create_batch([wordIds], [[]])
    return batch


if __name__ == '__main__':

    data_txt = 'data/data.txt'
    batch_size = 128

    # 得到分词后的sources和targets
    data = load_and_cut_data(data_txt)

    # 根据sources和targets创建词典，并映射
    data, word_to_id, id_to_word = create_dic_and_map(data)
    batches = get_batches(data, batch_size)

    temp = 0
    for nexBatch in batches:
        if temp == 0:
            print(len(nexBatch.encoder_inputs))
            print(len(nexBatch.encoder_inputs_length))
            print(len(nexBatch.decoder_targets_pre))
            print(len(nexBatch.decoder_targets_pre_length))
            print(len(nexBatch.decoder_targets_post))
            print(len(nexBatch.decoder_targets_post_length))
        temp += 1
