# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_
import nltk

padToken, unknownToken, goToken, eosToken = 0, 1, 2, 3


class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []


def load_data(filepath):
    '''
    加载数据
    :param filepath: 数据路径
    :return: data
    '''
    with open(filepath, 'r', encoding='UTF-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            data.append(line)
    return data


def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def createBatch(samples):
    batch = Batch()
    batch.encoder_inputs_length = [len(sample[0]) for sample in samples]
    batch.decoder_targets_length = [len(sample[1]) + 1 for sample in samples]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for sample in samples:
        # 将source进行反序并PAD值本batch的最大长度
        source = list(reversed(sample[0]))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

        # 将target进行PAD，并添加END符号
        target = sample[1]
        pad = [padToken] * (max_target_length - len(target) - 1)
        eos = [eosToken] * 1
        batch.decoder_targets.append(target + eos + pad)
        # batch.decoder_targets.append(target + pad)

    return batch


def getBatches(processed_data, batch_size):
    batches = []
    data_len = len(processed_data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield processed_data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        # samples有batchs_size行，每行是一个QA
        batch = createBatch(samples)
        batches.append(batch)
    return batches


def process_all_data(data):
    '''
    得到输入和输出的字符映射表
    :param data: 原始数据，内容为汉字
    :return: processed_data: 映射后的数据，内容为数字
             word_to_id: 字典，数字到数字的转换
             id_to_word: 字典，数字到汉字的转换
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_words = list(set([character for line in data for subline in line for character in subline]))
    id_to_word = {idx: word for idx, word in enumerate(special_words + set_words)}
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    # 将每一行转换成字符id的list
    processed_data = [[[word_to_id.get(word, word_to_id['<UNK>'])
                   for word in subline] for subline in line] for line in data]

    return processed_data, word_to_id, id_to_word


def sentence2enco(sentence, word2id):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    if sentence == '':
        return None
    # 分词
    # tokens = nltk.word_tokenize(sentence)
    tokens = ''.join(sentence.split())

    # 将每个单词转化为id
    wordIds = []
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
    # 调用createBatch构造batch
    batch = createBatch([[wordIds, []]])
    return batch


if __name__ == '__main__':
    filepath = 'data/data.txt'
    batch_size = 70
    data = load_data(filepath)
    processed_data, word_to_id, id_to_word = process_all_data(data)  # 根据词典映射
    batches = getBatches(processed_data, batch_size)

    temp = 0
    for nexBatch in batches:
        if temp == 0:
            print(len(nexBatch.encoder_inputs))
            print(len(nexBatch.encoder_inputs_length))
            print(nexBatch.decoder_targets)
            print(nexBatch.decoder_targets_length)
        temp += 1










