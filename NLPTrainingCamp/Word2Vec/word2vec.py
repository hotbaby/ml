# encoding: utf8

import os
import sys
import time
import gensim
import numpy as np
import pandas as pd
import multiprocessing
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

# BASEDIR = os.path.dirname(os.path.basename(__file__))
DATA_DIR = os.path.expanduser('~/Datasets')
wordsim353_data_path = os.path.join(DATA_DIR, 'wordsim353.csv') # noqa


class Sentence(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, encoding="utf8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                tokenized_line = ' '.join(gensim.utils.tokenize(line))
                is_alpha_word_line = [word for word in tokenized_line.lower().split() if word.isalpha()]
                yield is_alpha_word_line


def word2vec(data_path, size=100, window=5, min_count=10, binary=False, **kwargs):
    """
    Work2Vec

    :param data_path, 语料数据路径
    :param size, 向量维度
    :param window, 窗口大小
    :param min_count, 训练最小词频
    :param binary, 词向量是否为二进制
    :return word_vector 词向量
    """
    sentences = Sentence(data_path)
    model = gensim.models.Word2Vec(sentences, size=size, window=window, min_count=min_count,
                                   workers=multiprocessing.cpu_count())

    model.save('embedding/word2vec_model_%s_%s' % (size, window))
    model.wv.save_word2vec_format('embedding/word2vec_%s_%s.wv' % (size, window), binary=binary)

    return model.wv


def spearman_correlation(R, S): # noqa
    """
    Spearman相关系数.
    """
    # 转化成rank
    R = np.argsort(R)   # noqa
    S = np.argsort(S)   # noqa

    length = len(R)
    r_mean = np.mean(R)
    s_mean = np.mean(S)

    num1 = np.sum([(R[i] - r_mean) * (S[i] - s_mean) for i in range(length)])
    num2 = np.sqrt(np.sum([(R[i] - r_mean) ** 2 for i in range(length)]) *
                   np.sum([(S[i] - s_mean) ** 2 for i in range(length)]))

    return num1 / num2


def evaluate_similarity(wv):
    """
    评估word2vec相似性.
    :param wv: 词向量
    :return spearman系数
    """

    def similarity(w1, w2):
        """计算余弦相似度"""
        w1 = w1.lower()
        w2 = w2.lower()

        # 处理OOV情况
        if w1 not in wv or w2 not in wv:
            print('w1[%s] or w2[%s] out of vocabulary' % (w1, w2))
            return 0

        return cosine_similarity(wv[w1].reshape(1, -1), wv[w2].reshape(1, -1))

    # 读取wordsim353
    word_sim_353 = pd.read_csv(wordsim353_data_path)
    word_sim_353['wv'] = word_sim_353.apply(lambda x: similarity(x[0], x[1]), axis=1)

    # 计算spearman系数
    return spearman_correlation(word_sim_353.iloc[:, 2], word_sim_353['wv'])


if __name__ == '__main__':
    corpus_data_path = os.path.join(DATA_DIR, 'wiki.txt')
    print('start train word2vec, corpus: %s' % corpus_data_path)
    tm = time.time()
    wv = word2vec(corpus_data_path)
    print('train word2vec done. elapse %.3f seconds' % (time.time() - tm))

    wv = KeyedVectors.load_word2vec_format('./embedding/word2vec_100_5.wv')
    spearman_score = evaluate_similarity(wv)
    print('spearman score: %.2f' % spearman_score)
