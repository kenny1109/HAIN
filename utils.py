from sklearn import metrics
import numpy as np
import random
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from data.data_preprocess import LoadData
from data.data_preprocess import DataProcess


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.where(y_prob > 0, 1, 0)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'micro-f1' in list_metrics:
        output['micro-f1'] = metrics.f1_score(y_true, y_pred, average='micro')
    if 'micro_precision' in list_metrics:
        output['micro_precision'] = metrics.precision_score(y_true, y_pred, average='micro')
    if 'micro_recall' in list_metrics:
        output['micro_recall'] = metrics.recall_score(y_true, y_pred, average='micro')
    if 'hamming' in list_metrics:
        try:
            output['hamming'] = metrics.hamming_loss(y_true, y_pred)
        except ValueError:
            output['hamming'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.multilabel_confusion_matrix(y_true, y_pred))
        # print(metrics.classification_report(y_true, y_pred))
    return output

# class MyData(data.Dataset):
#     def __init__(self, root, state='train'):
#         self.root = root
#         with open(root, 'rb') as f:
#             [self.train_x, self.train_y,
#              self.val_x, self.val_y,
#              self.test_x, self.test_y,
#              self.word2index, self.label2index,
#              self.n_words, self.n_labels,
#              self.index2word, self.index2label
#              ] = pickle.load(f)
#         if state is 'train':
#             self.feature = self.train_x[:]
#             self.label = self.train_y[:]
#         elif state is 'test':
#             self.feature = self.test_x
#             self.label = self.test_y
#         else:
#             self.feature = self.val_x
#             self.label = self.val_y

#     def __getitem__(self, item):
#         feature = self.feature[item]
#         label = self.label[item]
#         return feature, label

#     def __len__(self):
#         return len(self.feature)


# def collate_func(batch):
#     feature = []
#     label = []
#     n_labels = 33
#     # start = time.time()
#     for i, j in batch:
#         feature.append(i)
#         label.append(j)
#     feature, sentence_sq = DataProcess.pad_sentence(feature)
#     feature, uttr_sq, sentence_sq = DataProcess.pad_utterance(feature, sentence_sq)
#     label = DataProcess.change_label_shape(label, n_labels)
#     # sentence_sq = DataProcess.pad_sq(sentence_sq)
#     bt = []
#     bt.append(torch.LongTensor(feature))
#     bt.append(torch.FloatTensor(label))

#     bt.append(torch.BoolTensor(sentence_sq))
#     bt.append(torch.BoolTensor(uttr_sq))
#     # end = time.time()
#     # print('cost time {}'.format(end - start))
#     return bt

class MyData(data.Dataset):
    def __init__(self, root, state='train', batch_size=64):
        self.root = root
        with open(root, 'rb') as f:
            [self.train_x, self.train_y,
             self.val_x, self.val_y,
             self.test_x, self.test_y,
             self.word2index, self.label2index,
             self.n_words, self.n_labels,
             self.index2word, self.index2label
             ] = pickle.load(f)
        feature = []
        sentence_sq = []
        uttr_sq = []
        if state is 'train':
            for i in range(0, len(self.train_y), batch_size):
                f, s = DataProcess.pad_sentence(self.train_x[i:i+batch_size])
                f, u, s = DataProcess.pad_utterance(f, s)
                feature.extend(f)
                sentence_sq.extend(s)
                uttr_sq.extend(u)
            self.label = DataProcess.change_label_shape(self.train_y, self.n_labels)
        elif state is 'train_valid':
            x = self.train_x+self.val_x
            y = self.train_y+self.val_y
            for i in range(0, len(y), batch_size):
                f, s = DataProcess.pad_sentence(x[i:i+batch_size])
                f, u, s = DataProcess.pad_utterance(f, s)
                feature.extend(f)
                sentence_sq.extend(s)
                uttr_sq.extend(u)
            self.label = DataProcess.change_label_shape(y, self.n_labels)
        elif state is 'test':
            for i in range(0, len(self.test_y), batch_size):
                f, s = DataProcess.pad_sentence(self.test_x[i:i + batch_size])
                f, u, s = DataProcess.pad_utterance(f, s)
                feature.extend(f)
                sentence_sq.extend(s)
                uttr_sq.extend(u)
            self.label = DataProcess.change_label_shape(self.test_y, self.n_labels)
        else:
            for i in range(0, len(self.val_y), batch_size):
                f, s = DataProcess.pad_sentence(self.val_x[i:i + batch_size])
                f, u, s = DataProcess.pad_utterance(f, s)
                feature.extend(f)
                sentence_sq.extend(s)
                uttr_sq.extend(u)
            self.label = DataProcess.change_label_shape(self.val_y, self.n_labels)
        self.feature = feature
        self.sentence_sq = sentence_sq
        self.uttr_sq = uttr_sq

    def __getitem__(self, item):
        feature = self.feature[item]
        label = self.label[item]
        sentence_sq = self.sentence_sq[item]
        uttr_sq = self.uttr_sq[item]
        return feature, label, sentence_sq, uttr_sq

    def __len__(self):
        return len(self.feature)


def collate_func(batch):
    feature = []
    label = []
    sentence_sq = []
    uttr_sq = []

    # n_labels = 33
    # start = time.time()
    # for i, j in batch:
    #     feature.append(i)
    #     label.append(j)
    # feature, sentence_sq = DataProcess.pad_sentence(feature)
    # feature, uttr_sq, sentence_sq = DataProcess.pad_utterance(feature, sentence_sq)
    # label = DataProcess.change_label_shape(label, n_labels)
    # sentence_sq = DataProcess.pad_sq(sentence_sq)

    for i, j, k, l in batch:
        feature.append(i)
        label.append(j)
        sentence_sq.append(k)
        uttr_sq.append(l)

    bt = []
    bt.append(torch.LongTensor(feature))
    bt.append(torch.FloatTensor(label))

    bt.append(torch.BoolTensor(sentence_sq))
    bt.append(torch.BoolTensor(uttr_sq))
    # end = time.time()
    # print('cost time {}'.format(end - start))
    return bt


def adjust_learning_rate(optm, d_model, step_num, warmup_steps=2000):
    lr = (d_model ** (-0.4)) * min(step_num ** (-0.7), step_num * (warmup_steps ** (-1.7)))
    for param_group in optm.param_groups:
        param_group['lr'] = lr
    return lr


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def attention_visualization(attn_map):
    sns.set()
    return sns.heatmap(attn_map)

