# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import jieba

padToken, unknownToken, goToken, eosToken = 0, 1, 2, 3


class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []


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


def create_dic_and_map(sources, targets):
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

    # 得到每次词语的使用频率
    word_dic = {}
    for line in (sources + targets):
        for character in line:
            word_dic[character] = word_dic.get(character, 0) + 1

    # 去掉使用频率为1的词
    # word_dic_new = [k for k, v in word_dic.items() if v > 1]
    # word_dic_new = []
    # for key, value in word_dic.items():
    #     if value > 1:
    #         word_dic_new.append(key)

    # 不去掉频率为1的词
    word_dic_new = [k for k, _ in word_dic.items()]

    # 将字典中的汉字/英文单词映射为数字
    id_to_word = {idx: word for idx, word in enumerate(special_words + word_dic_new)}
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    # 将sources和targets中的汉字/英文单词映射为数字
    sources_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in sources]
    targets_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in targets]

    return sources_data, targets_data, word_to_id, id_to_word


def createBatch(sources, targets):
    batch = Batch()
    batch.encoder_inputs_length = [len(source) for source in sources]
    # len(target) + 1 because of one <EOS>
    batch.decoder_targets_length = [len(target) + 1 for target in targets]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for source in sources:
        # 将source进行反序并PAD
        source = list(reversed(source))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

    for target in targets:
        # 将target进行PAD，并添加EOS符号
        pad = [padToken] * (max_target_length - (len(target) + 1))
        eos = [eosToken] * 1
        batch.decoder_targets.append(target + eos + pad)

    return batch


def getBatches(sources_data, targets_data, batch_size):
    data_len = len(sources_data)

    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield sources_data[i:min(i + batch_size, data_len)], targets_data[i:min(i + batch_size, data_len)]

    batches = []
    for sources, targets in genNextSamples():
        batch = createBatch(sources, targets)
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
    print(wordIds)
    # 调用createBatch构造batch
    batch = createBatch([wordIds], [[]])
    return batch


if __name__ == '__main__':

    sources_txt = 'data/sources.txt'
    targets_txt = 'data/targets.txt'
    keep_rate = 0.6
    batch_size = 128

    # 得到分词后的sources和targets
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射
    sources_data, targets_data, word_to_id, id_to_word = create_dic_and_map(sources, targets)
    batches = getBatches(sources_data, targets_data, batch_size)

    temp = 0
    for nexBatch in batches:
        if temp == 0:
            print(len(nexBatch.encoder_inputs))
            print(len(nexBatch.encoder_inputs_length))
            print(nexBatch.decoder_targets)
            print(nexBatch.decoder_targets_length)
        temp += 1
