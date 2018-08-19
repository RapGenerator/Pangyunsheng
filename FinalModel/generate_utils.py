import math
import json
import operator

from pypinyin import Style, lazy_pinyin


class Beam(object):
    beams = []
    extending_beams = []
    stop_index = None
    encourage_index = []
    discourage_index = []

    class beam_node:
        def __init__(self, prob, index, state, stop_index, force_stop=False):
            self.prob = prob
            self.log_prob = math.log2(prob)
            self.index = index
            self.state = state
            self.stopped = (self.index == stop_index)
            self.node_score = self.log_prob
            if force_stop:
                self.stopped = True

    def __init__(self, width=10, stop_index=35003, index2word=None, start_len=None, ensure_len=False, max_len=12,
                 target_long=5):
        self.beam_width = width
        self.stop_index = stop_index
        self.index2word = index2word
        self.start_len = start_len
        self.ensure_len = ensure_len
        self.max_len = max_len

        # A reward for a sentence approaching target_long length, usually between 0.7-1.0.\
        # The length of the sentence user want
        self.reward_factor = 0.7
        self.target_long = target_long

        self.beams = []
        start_node = self.beam_node(prob=1, index=-1, state=None, stop_index=self.stop_index)
        self.beams.append([start_node])

    def add_prob(self, prob, index, state, beam_num):
        beam = self.beams[beam_num]
        force_stop = len(beam) > self.max_len
        node = self.beam_node(prob, index, state, self.stop_index, force_stop, )
        new_beam = beam + [node]
        self.extending_beams.append(new_beam)

    # empty extending_beams, keep the beams of top beam width by probability
    def shrink_beam(self):
        checking = self.extending_beams
        for beam in self.beams:
            if beam[-1].stopped:
                checking.append(beam)
        if len(checking) <= self.beam_width:
            self.beams = checking
            self.extending_beams = []
            return

        to_order = {}
        for i in range(len(checking)):
            prob = self.get_beam_score(checking[i])
            to_order[i] = prob
        to_order = sorted(to_order.items(), key=operator.itemgetter(1), reverse=True)
        to_order = to_order[:self.beam_width]
        self.beams = [checking[i] for (i, value) in to_order]
        self.extending_beams = []

    def get_best(self):
        best_beam = self.beams[0]
        best_prob = self.get_beam_score(self.beams[0])
        found = False
        for beam in self.beams:
            if self.ensure_len and self.get_beam_word_len(beam) != self.target_long:
                continue
            prob = self.get_beam_score(beam)
            if best_prob < prob:
                best_beam = beam
                best_prob = prob
                found = True
        if self.ensure_len and not found:
            for beam in self.beams:
                prob = self.get_beam_score(beam)
                if best_prob < prob:
                    best_beam = beam
                    best_prob = prob
        best_index = []
        for node in best_beam:
            best_index.append(node.index)
        return best_index[1:], best_beam[1:]

    def check_finished(self):
        for beam in self.beams:
            if not beam[-1].stopped:
                return False
        return True

    def get_beam_score(self, beam):
        prob = 0
        for node in beam:
            prob += node.node_score
        s_len = self.get_beam_word_len(beam)
        prob /= math.pow(s_len, 1)
        prob *= math.pow(abs(s_len-self.target_long)+1, self.reward_factor)
        return prob

    def get_beam_word_len(self, beam):
        s_len = 0
        for node in beam:
            if self.index2word.get(node.index) is not None and node.index != self.stop_index:
                s_len += len(self.index2word[node.index])
        s_len += self.start_len
        return s_len


class RhymeChecker(object):
    used = []

    def __init__(self, rhyme_path):
        with open(rhyme_path, 'r') as inputfile:
            self.rhyme_index = json.load(inputfile)

    def get_yunmu(self, word):
        ym = []
        for c in word:
           yunmu = lazy_pinyin(c, style=Style.FINALS, strict=False)
           ym.append(yunmu[0])
        return ym

    def check_two(self, tagart_word, judge_word, rhyme_mode):
        tagart_ym = self.get_yunmu(tagart_word)
        judge_ym = self.get_yunmu(judge_word)
        if rhyme_mode == 1:
            if tagart_ym[0] == judge_ym[-1]:
                return True
            else:
                return False

    def find_match(self, word, choices, rhyme_mode):
        ret = []
        for choice in choices:
            if choice in self.used:
                continue
            if rhyme_mode == 1:
                if self.check_two(word, choice, rhyme_mode):
                    self.used.append(choice)
                    ret.append(choice)
                    break
                else:
                    continue

            elif rhyme_mode == 2:
                pass
            else:
                pass
        return ret[0]

    @staticmethod
    def check_substring(word1, word2):
        result = word1 in word2 or word2 in word1
        return result


def sort_prob(prob):
    """
    Ranking index by probability from high to low
    :param probs:
    :return:
    """
    mydict = dict(enumerate(prob))
    sorted_index = sorted(mydict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_index


def sort_word_by_prob(int_to_word, prob):
    """
    Ranking word by probability from high to low
    :param int_to_word: dict of 'int_to_word'
    :param prob: predict probability of each word
    :return:
    """
    sorted_index_pro = sort_prob(prob)
    sorted_index = [f for (f, s) in sorted_index_pro]
    sorted_word = [int_to_word[num] for num in sorted_index]
    return sorted_word


def get_sort_word_by_prob(int_to_word, prob):
    """
    从distribution中得到除'<UNK>'，'<PAD>'，'<GO>'，'<EOS>'外概率最大的词
    :param int_to_word: dict of 'int_to_word'
    :param prob: predict probability of each word
    :return: get_word: 概率最大的词
    """
    get_word = None
    sorted_word = sort_word_by_prob(int_to_word, prob)
    for w in sorted_word:
        if w == '<UNK>' or w == '<PAD>' or w == '<GO>' or w == '<EOS>':
            continue
        get_word = w
        break
    return get_word


def choice_rhyme_word(last_word, sorted_word, num, random, used_words):
    """
    在sorted_word中从前往后找到跟last_word的韵脚相同的第一个词
    全部遍历后仍找不到的话，就随机返回一个词
    :param last_word: 传进来的就是需要全部押韵的部分
    :param sorted_word: 传进来的是按照概率从大到小排列的词，这里面所有词的长度是一定不小于last_word的长度的
    :return:
    """
    if random is True:
        return sorted_word[0:num]
    words = []
    last_word_ym = lazy_pinyin(last_word, style=Style.FINALS, strict=False)
    last_word_ym.reverse()
    find_word = 0
    for word in sorted_word:
        word_ym = lazy_pinyin(word, style=Style.FINALS, strict=False)
        word_ym.reverse()
        for i in range(len(last_word_ym)):
            if last_word_ym[i] != word_ym[i]:
                break
            if i == len(last_word_ym)-1:
                find_word = 1
        if find_word == 1:
            if word not in used_words:
                words.append(word)
            find_word = 0
            if len(words) >= num:
                return words
    # words 候选词的个数不够20个，那就有多少返回去多少
    if len(words) > 0:
        return words
    # words 一个满足条件的候选词都没有，那就返回前top num
    else:
        return sorted_word[0:num]


# def get_replace(word_to_int, int_to_word, prob, last_word, rhyme_checker, limit=2, rhyme_mode=1, used_words=[]):
def get_replace(last_word, limit, prob, int_to_word, word_to_int, num, random=False, used_words=[]):
    """
    According to the rhyme words of the preceding sentence
    and the probability distribution of the first word  of this sentence,
    return the rhyme words of this sentence.
    :param word_to_int:
    :param int_to_word:
    :param prob:
    :param last_word:the rhyme words of the preceding sentence
    :param limit:The len of candidate rhyme words is 2
    :param rhyme_mode: The last one word rhyme
    :param rhyme_checker:
    :return:Rhyme word in this sentence
    """
    # sorted_word = sort_word_by_prob(int_to_word, prob)
    # if limit == 2:
    #     sorted_word = [w for w in sorted_word if len(w) == limit]
    #     this_word = rhyme_checker.find_match(last_word, sorted_word, rhyme_mode)
    #     replaced = word_to_int[this_word]
    # else:
    #
    #     pass
    # return this_word, replaced

    result_len = max(len(last_word), limit)
    prob_dict = dict(enumerate(prob))
    sorted_index_prob = sorted(prob_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_index = [f for (f, s) in sorted_index_prob]
    sorted_word = [int_to_word[num] for num in sorted_index]
    sorted_word = [w for w in sorted_word if len(w) >= result_len]
    # last_word是需要押韵的部分。sorted_word是按照概率从大到小的顺序的候选押韵词
    # 得到last_word和sorted_word的韵母，看是不是从右开始算的包含关系，
    # 还要检查之前有没有出现过，出现过的词是不可以在被选择的

    rhyme_words = choice_rhyme_word(last_word, sorted_word, num, random, used_words)

    rhyme_words_index = []
    for rhyme_word in rhyme_words:
        rhyme_words_index.append(word_to_int[rhyme_word])
    return rhyme_words, rhyme_words_index


def get_next_sentence_rhyme_word(last_word,
                                 rhyme_mode,
                                 result_rhyme_word_len,
                                 prob,
                                 int_to_word,
                                 word_to_int,
                                 num=20,
                                 random=False,
                                 used_words=[]):
    """

    :param last_word: last sentence 分词后的最后一个词，
    :param rhyme_mode: 默认情况是rhyme_mode = len（last sentence）
        可以设置rhyme_mode<len（last sentence）
        但如果用户设置rhyme_mode>len（last sentence）那也只能做len（last sentence）的押韵
        即实际的押韵方式是min（last_word，rhyme_mode）
    :param result_rhyme_word_len: 默认值是2
        用户可以设置返回的押韵词的字数至少是多大的
        实际返回的押韵词结果的字数的大小是 max(result_rhyme_word_len, min（last_word，rhyme_mode)
    :param prob:
    :param int_to_word:
    :param word_to_int:
    :param used_rhyme_words:
    :param num: 控制返回候选词的最大个数
    :param random:默认是false即根据last_word的韵脚生成，True的时候指，不根据last_word，返回一个list
    :return: 返回一个从大到小符合要求的list 词表
    """
    rhyme_mode = min(len(last_word), rhyme_mode)
    result_rhyme_word_len = max(result_rhyme_word_len, rhyme_mode)
    last_word = last_word[-rhyme_mode:]

    rhyme_words, rhyme_words_index = get_replace(last_word=last_word,
                                                 limit=result_rhyme_word_len,
                                                 prob=prob,
                                                 int_to_word=int_to_word,
                                                 word_to_int=word_to_int,
                                                 num=num,
                                                 random=random,
                                                 used_words=used_words)
    return rhyme_words, rhyme_words_index
