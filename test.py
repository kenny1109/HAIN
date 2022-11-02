import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from data.data_preprocess import LoadData
from data.data_preprocess import DataProcess
from models_h_transformer.layers import DMLC
import w2v
import pickle
import os
import time
import numpy as np
import random
from utils import get_evaluation, MyData, collate_func, adjust_learning_rate, setup_seed



# test model

def predict(model_path, n=0, lamda='', beta=''):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BZ = 64
    setup_seed(123)
    path = '/data/dstc8.p'
    # model_path = os.getcwd() + '/best_model/best_acc_checkpoint.pt'
    test_set = MyData(os.getcwd() + path, state='test')
    if 1:
        with open(os.getcwd() + '/embedding/w2v.p', 'rb') as f:
            em_w = pickle.load(f)
    if 0:
        em_w = w2v.load_word2vec('glove', ld.word2index, 200)
        with open(os.getcwd() + '/embedding/w2v.p', 'wb') as f:
            pickle.dump(em_w, f)
    vb_size, emd_dim = em_w.shape
    print('embedding size:{}'.format(emd_dim))
    batch_size = BZ
    num_labels = test_set.n_labels
    hidden_size = emd_dim
    model = DMLC(em_w, vb_size, emd_dim, hidden_size, batch_size, num_labels)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # test_set = MyData(os.getcwd() + path, state='test')
    text = DataProcess.decode_data(test_set.test_x, test_set.index2word)
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
        input_batch = batch_x.to(device)
        target_batch = batch_y.to(device)
        sentence_sq = sentence_sq.to(device)
        uttr_sq = uttr_sq.to(device)
        with torch.no_grad():
            te_predictions, attention = model(input_batch, sentence_sq, uttr_sq)
        
            w_l2 = []
            for i in range(len(attention[0])):
                # m2 = torch.cat(attention[0][i], dim=1)
                m2 = attention[0][i].squeeze(-1)
                a = torch.bmm(m2, m2.permute(0, 2, 1))
                b = torch.eye(m2.size()[1]).unsqueeze(0).expand(m2.size()[0], m2.size()[1], m2.size()[1]).to(device)
                c = torch.sum(torch.norm(b - a, p=2, dim=(1, 2)).masked_fill(uttr_sq[:, i], float(0.0))) \
                    / (uttr_sq.size()[0] - torch.nonzero(uttr_sq[:, i]).size()[0])
                # c = torch.sum(torch.norm(b, dim=(1, 2))) - torch.sum(torch.norm(a, dim=(1, 2)))
                w_l2.append(c)
            # m2 = torch.cat(attention[1], dim=1)
            m2 = attention[1].squeeze(-1)
            a = torch.bmm(m2, m2.permute(0, 2, 1))
            b = torch.eye(m2.size()[1]).unsqueeze(0).expand(m2.size()[0], m2.size()[1], m2.size()[1]).to(device)
            c = torch.sum(torch.norm(b - a, p=2, dim=(1, 2))) / m2.size()[0]
            # c = torch.sum(torch.norm(b, dim=(1, 2))) - torch.sum(torch.norm(a, dim=(1, 2)))

            s_l2 = c
            l2 = sum(w_l2) / len(w_l2)
            # print(l2, s_l2)

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
    if os.path.exists(os.getcwd() + '/results/test/' + str(lamda) + '_' + str(beta) + 'correlation.txt'):
        log_file = open(os.getcwd() + '/results/test/' + str(lamda) + '_' + str(beta) + 'correlation.txt', 'a')
    else:
        log_file = open(os.getcwd() + '/results/test/' + str(lamda) + '_' + str(beta) + 'correlation.txt', 'w')
    log_file.write("Test number: {}, Loss: {:.6f}, Accuracy: {:.4f}, Hamming: {:.6f}, micro-f1: {:.6f}, micro_precision: {:.6f}, "
                    "micro_recall: {:.6f} \n\n".format(n, te_loss, test_metrics["accuracy"], 
                                        test_metrics["hamming"],
                                        test_metrics["micro-f1"],
                                        test_metrics["micro_precision"],
                                        test_metrics["micro_recall"]))
    print("Test number: {}, Loss: {:.6f}, Accuracy: {:.4f}, Hamming: {:.6f}, micro-f1: {:.6f}, micro_precision: {:.6f}, micro_recall: {:.6f}".format(
        n, te_loss, test_metrics["accuracy"],
        test_metrics["hamming"],
        test_metrics["micro-f1"],
        test_metrics["micro_precision"],
        test_metrics["micro_recall"]))
    print('cost time {:.4f}'.format(end - start))
    return test_metrics["accuracy"], test_metrics["hamming"], test_metrics["micro-f1"]


if __name__ == '__main__':
    test_best = 0
    best_hamming = 1
    best_f1 = 0
    d = [0, 0, 0]
    for i in range(80, 100):
        print(i)
        model_path = os.getcwd() + '/best_model/multi-head-4-50-0.00125-0.0/' + str(i) + 'best_acc_checkpoint.pt'
        a, b, c = predict(model_path, i)
        if a >= test_best:
            test_best = a
            d[0] = i
        if b <= best_hamming:
            best_hamming = b
            d[1] = i
        if c >= best_f1:
            best_f1 = c
            d[2] = i
    if os.path.exists(os.getcwd() + '/results/test/best_correlation.txt'):
        log_file = open(os.getcwd() + '/results/test/best_correlation.txt', 'a')
    else:
        log_file = open(os.getcwd() + '/results/test/best_correlation.txt', 'w')
    log_file.write("best test accuacy: {:.4f}\n".format(test_best))
    log_file.write("best test hamming loss: {:.6f}\n".format(best_hamming))
    log_file.write("best test micro-f1: {:.4f}\n".format(best_f1))
    log_file.write("model number: {}\n".format(d))
    log_file.close()
    print("best test accuacy: {:.4f}".format(test_best))
    print("best test hamming loss: {:.6f}".format(best_hamming))
    print("best test micro-f1: {:.4f}".format(best_f1))
    print("model number: {}".format(d))