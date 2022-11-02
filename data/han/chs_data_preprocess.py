import json
import time
import re
import os
import pickle
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
# import spacy
from nltk.tokenize import word_tokenize
import jieba


class DataProcess:
    def __init__(self):
        self.word2index = {'<pad>': 0, '<unk>': 1}
        self.word2count = {'<pad>': 0, '<unk>': 0}
        self.index2word = {0: '<pad>', 1: '<unk>'}
        self.n_words = 2
        self.label2index = {}

        self.label2count = {}
        self.index2label = {}
        self.n_labels = 0

    @staticmethod
    def process_ori_text():
        path = os.getcwd()
        train = pd.read_csv(path + '/../chs_data_train.csv', encoding='utf-8')
        test = pd.read_csv(path + '/../chs_data_test.csv', encoding='utf-8')

        for i in range(len(train['text'])):
            t = train['text'][i].split('\n')
            temp = []
            for j in range(len(t)):
                k = t[j].split('。坐席')
                k[1] = '坐席' + k[1]
                temp.extend(k)
            train['text'][i] = '\n'.join(temp)
            print(train['text'][i])

        for i in range(len(test['text'])):
            t = test['text'][i].split('\n')
            temp = []
            for j in range(len(t)):
                k = t[j].split('。坐席')
                k[1] = '坐席' + k[1]
                temp.extend(k)
            test['text'][i] = '\n'.join(temp)
            print(test['text'][i])

        train.to_csv(path + '/chs_data_train_han' + '.csv', header=True, index=False)
        test.to_csv(path + '/chs_data_test_han' + '.csv', header=True, index=False)
        return

    def add_word_from_sentence(self, doc):
        if isinstance(doc, list):
            for sentence in doc:
                sentence = self.clear_str(sentence)
                for word in sentence.split(' '):
                    if word is not '':
                        self.add_word(word)
        else:
            raise ValueError('add_word_from_sentence: input format err!')

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def build_vocab(self, doc):
        if isinstance(doc, list):
            for sentences in doc:
                if isinstance(sentences, list):
                    for sentence in sentences:
                        for word in sentence:
                            self.add_word(word)
                else:
                    raise ValueError('build_vocab: input format err!')
        else:
            raise ValueError('build_vocab: input format err!')

    def add_label(self, name):
        if name not in self.label2index:
            self.label2index[name] = self.n_labels
            self.label2count[name] = 1
            self.index2label[self.n_labels] = name
            self.n_labels += 1
        else:
            self.label2count[name] += 1

    def build_label(self, label_set):
        if isinstance(label_set, list):
            for label in label_set:
                if isinstance(label, list):
                    for name in label:
                        self.add_label(name)
                else:
                    raise ValueError('build_label: input format err!')
        else:
            raise ValueError('build_label: input format err!')

    def word_segmentation(self, doc):
        if isinstance(doc, list):
            encode = []
            for text in doc:
                if isinstance(text, str):
                    encode_text = []
                    sentences = text.split('\n')[:-1]
                    for sentence in sentences:
                        sentence = sentence.lower()
                        word_token = jieba.lcut(sentence, cut_all=False)
                        if len(word_token) == 0:
                            raise ValueError('word_segmentation: word_token format err!')
                        # sentence = self.word2token(sentence)
                        # word_token = []
                        # for token in sentence:
                        #     word_token.append(token.text)
                        encode_text.append(word_token)
                else:
                    raise ValueError('word_segmentation: input format err!')
                encode.append(encode_text)
            return encode

        else:
            raise ValueError('word_segmentation: input format err!')

    # 按批次中最大的序列长度进行处理,且序列已经分词
    @staticmethod
    def pad_sentence(doc, pad=0, max_length=250):

        if isinstance(doc, list):
            t = []
            sq = []
            for sentences in doc:
                for x in sentences:
                    t.append(len(x))
            sequence_length = min(max(t), max_length)
            padded_doc = []
            for sentences in doc:
                padded_sentences = []
                sq_tmp = []
                for i in range(len(sentences)):
                    sentence = sentences[i]
                    if isinstance(sentences, list):
                        if len(sentence) < max_length:
                            if len(sentence) == 0:
                                raise ValueError('pad_sentence: sentence format err!')
                            num_padding = sequence_length - len(sentence)
                            sq_tmp.append([0] * len(sentence) + [1] * num_padding)
                            new_sentence = sentence + [pad] * num_padding
                        else:
                            new_sentence = sentence[:max_length]
                            sq_tmp.append([0] * max_length)
                        padded_sentences.append(new_sentence)
                    else:
                        raise ValueError('pad_sentence: input format err!')
                padded_doc.append(padded_sentences)
                sq.append(sq_tmp)
            return padded_doc, sq
        else:
            raise ValueError('pad_sentence: input format err!')

    @staticmethod
    def pad_utterance(uttr, ssq, pad=0, unk=1):
        sequence_length = max(len(x) for x in uttr)
        # print(sequence_length)
        padded_uttr = []
        padded_ssq = []
        sq = []
        for sentences, s in zip(uttr, ssq):
            num_padding = sequence_length - len(sentences)
            sq.append([0] * len(sentences) + [1] * num_padding)
            for j in range(0, num_padding):
                sentences.append([pad] * len(sentences[0]))
                # s.append([1] + [0] * (len(s[0]) - 1))
                s.append([0] * (len(s[0])))
            padded_uttr.append(sentences)
            padded_ssq.append(s)
        return padded_uttr, sq, padded_ssq

    @staticmethod
    def pad_sq(sq, pad=0):
        sequence_length = max(len(x) for x in sq)
        padded_sq = []
        for s in sq:
            num_padding = sequence_length - len(s)
            s = s + [pad] * num_padding
            padded_sq.append(s)
        return padded_sq

    def encode_data(self, data, vocabulary):
        if isinstance(data, list):
            encode = []
            for sentences in data:
                if isinstance(sentences, list):
                    encode_text = []
                    for sentence in sentences:
                        encode_sentence = []
                        for word in sentence:
                            encode_sentence.append(vocabulary[word] if word in vocabulary else self.word2index['<unk>'])
                        encode_text.append(encode_sentence)
                else:
                    raise ValueError('encode_data: input format err!')
                encode.append(encode_text)
            return encode
        else:
            raise ValueError('encode_data: input format err!')

    @staticmethod
    def encode_label(labels, vocabulary):
        if isinstance(labels, list):
            encode = []
            for label in labels:
                encode_l = []
                for l in label:
                    encode_l.append(vocabulary[l] if l in vocabulary else print('out of label set!'))
                encode.append(encode_l)
            return encode
        else:
            raise ValueError('encode_label: input format err!')

    @staticmethod
    def change_label_shape(labels, n_labels):
        if isinstance(labels, list):
            label_vector = []
            for label in labels:
                lb = [0] * n_labels
                for n in label:
                    lb[n] = 1
                label_vector.append(lb)
            return label_vector
        else:
            raise ValueError('change_label_shape: input format err!')


class LoadData(DataProcess):
    def __init__(self, path1, path2, max_length=150, vocab_size=50000):
        super(LoadData, self).__init__()
        self.train_path = path1
        self.test_path = path2
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        self.rnd = None
        self.encode_train_x = None
        self.encode_train_y = None
        self.encode_val_x = None
        self.encode_val_y = None
        self.encode_test_x = None
        self.encode_test_y = None
        self.train_sq_len = None
        self.val_sq_len = None
        self.test_sq_len = None

    def load_data(self, load=False, save=True):
        # Load and preprocess data
        if load is False:
            train = pd.read_csv(os.getcwd() + self.train_path, encoding='utf-8')
            train_id = list(train['id'].values)
            self.train_x = list(train['text'].values)
            self.train_y = list(train['label'].str.split(' ').values)
            # print(train['label'].value_counts() / len(train['label']))

            test = pd.read_csv(os.getcwd() + self.test_path, encoding='utf-8')
            test_id = list(test['id'].values)
            self.test_x = list(test['text'].values)
            self.test_y = list(test['label'].str.split(' ').values)
            # print(test['label'].value_counts() / len(test['label']))

            self.train_x = self.word_segmentation(self.train_x)
            self.test_x = self.word_segmentation(self.test_x)
            self.build_vocab(self.train_x)
            self.build_label(self.train_y)
            self.encode_train_x = self.encode_data(self.train_x, self.word2index)
            self.encode_train_y = self.encode_label(self.train_y, self.label2index)
            self.encode_test_x = self.encode_data(self.test_x, self.word2index)
            self.encode_test_y = self.encode_label(self.test_y, self.label2index)
        else:
            with open(os.getcwd() + self.path, 'rb') as f:
                [self.encode_train_x, self.encode_train_y,
                 self.encode_val_x, self.encode_val_y,
                 self.encode_test_x, self.encode_test_y,
                 self.word2index, self.label2index,
                 self.n_words, self.n_labels,
                 self.index2word, self.index2label
                 ] = pickle.load(f)
        if save is True and load is False:

            p = self.train_path.split('/')
            p[-1] = 'telecom_data.p'
            p = '/'.join(p).strip()
            with open(os.getcwd() + p, 'wb') as f:
                pickle.dump((self.encode_train_x, self.encode_train_y,
                             self.encode_val_x, self.encode_val_y,
                             self.encode_test_x, self.encode_test_y,
                             self.word2index, self.label2index,
                             self.n_words, self.n_labels,
                             # self.index2word, self.index2label
                             ), f)
        # print(len(self.word2index))


if __name__ == '__main__':
    dp = DataProcess()
    # dp.process_ori_text()
    # exit()
    path1 = '/chs_data_train_han.csv'
    path2 = '/chs_data_test_han.csv'
    # path = '/data/dstc8.p'
    star = time.time()
    ld = LoadData(path1, path2)
    ld.load_data(load=False, save=True)
    end = time.time()
    print('cost time: {}'.format(end-star))
    # print(ld.change_label_shape(ld.encode_train_y))