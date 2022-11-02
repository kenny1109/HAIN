import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from data.cnn.data_preprocess import LoadData
from data.cnn.data_preprocess import DataProcess
from model_cnn.cnn import CNN
import w2v
import pickle
import os
import time
import numpy as np
import random
from utils import get_evaluation


class MyData(data.Dataset):
    def __init__(self, root, state='train'):
        self.root = root
        with open(root, 'rb') as f:
            [self.train_x, self.train_y,
             self.val_x, self.val_y,
             self.test_x, self.test_y,
             self.word2index, self.label2index,
             self.n_words, self.n_labels,
             # self.train_sq_len, self.test_sq_len
             ] = pickle.load(f)
        if state is 'train':
            self.feature = self.train_x[:]
            self.label = self.train_y[:]
        elif state is 'test':
            self.feature = self.test_x
            self.label = self.test_y
        else:
            self.feature = self.val_x
            self.label = self.val_y

    def __getitem__(self, item):
        feature = self.feature[item]
        label = self.label[item]
        return feature, label

    def __len__(self):
        return len(self.feature)


def collate_func(batch):
    feature = []
    label = []
    n_labels = 33
    # start = time.time()
    for i, j in batch:
        feature.append(i)
        label.append(j)
    feature, sentence_sq = DataProcess.pad_sentence(feature)
    feature, uttr_sq, sentence_sq = DataProcess.pad_utterance(feature, sentence_sq)
    label = DataProcess.change_label_shape(label, n_labels)
    # sentence_sq = DataProcess.pad_sq(sentence_sq)
    bt = []
    bt.append(torch.LongTensor(feature))
    bt.append(torch.FloatTensor(label))

    bt.append(torch.BoolTensor(sentence_sq))
    bt.append(torch.BoolTensor(uttr_sq))
    # end = time.time()
    # print('cost time {}'.format(end - start))
    return bt


def adjust_learning_rate(optm, d_model, step_num, warmup_steps=1200):
    lr = (d_model ** (-1)) * min(step_num ** (-0.65), step_num * (warmup_steps ** (-1.65)))
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


# test model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BZ = 64
    setup_seed(0)
    path = '/data/cnn/dstc8.p'
    model_path = os.getcwd() + '/best_model/cnn_best_acc_checkpoint.pt'
    if 1:
        with open(os.getcwd() + '/embedding/cnn_w2v.p', 'rb') as f:
            em_w = pickle.load(f)
    if 0:
        em_w = w2v.load_word2vec('glove', ld.word2index, 100)
        with open(os.getcwd() + '/embedding/cnn_w2v.p', 'wb') as f:
            pickle.dump(em_w, f)
    vb_size, emd_dim = em_w.shape
    print('embedding size:{}'.format(emd_dim))
    print('vocabulary size:{}'.format(vb_size))
    batch_size = BZ
    num_labels = 33
    model = CNN(em_w, vb_size, emd_dim, num_labels).to(device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    test_set = MyData(os.getcwd() + path, state='test')
    test_loader = data.DataLoader(
        dataset=test_set,  # torch TensorDataset format
        batch_size=BZ,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
        drop_last=False,
        collate_fn=collate_func
    )

    start = time.time()
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    loss_ls = []
    te_label_ls = []
    te_pred_ls = []
    for step, (batch_x, batch_y, sentence_sq, uttr_sq) in enumerate(test_loader):
        num_sample = len(batch_y)
        input_batch = batch_x.squeeze(1).to(device)
        target_batch = batch_y.to(device)
        with torch.no_grad():
            te_predictions = model(input_batch)
        te_loss = criterion(te_predictions, target_batch)
        loss_ls.append(te_loss * num_sample)
        te_label_ls.extend(target_batch.clone().cpu().unsqueeze(0))
        te_pred_ls.append(te_predictions.clone().cpu())

    te_loss = sum(loss_ls) / test_set.__len__()
    te_pred = torch.cat(te_pred_ls, 0)
    te_label = torch.cat(te_label_ls, 0)
    test_metrics = get_evaluation(te_label.numpy(), te_pred.numpy(), list_metrics=["accuracy",
                                                                                   "hamming",
                                                                                   "micro-f1",
                                                                                   "micro_precision",
                                                                                   "micro_recall"])
    end = time.time()
    print("Test Loss: {:.6f}, Accuracy: {:.4f}, Hamming: {:.6f}, micro-f1: {:.6f}, micro_precision: {:.6f}, micro_recall: {:.6f}".format(
        te_loss, test_metrics["accuracy"],
        test_metrics["hamming"],
        test_metrics["micro-f1"],
        test_metrics["micro_precision"],
        test_metrics["micro_recall"]))
    print('cost time {:.4f}'.format(end - start))


if __name__ == '__main__':
    main()
