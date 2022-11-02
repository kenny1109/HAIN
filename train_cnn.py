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
from tensorboardX import SummaryWriter


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
            self.feature, self.sentence_sq = DataProcess.pad_sentence(self.train_x)
            self.feature, self.uttr_sq, self.sentence_sq = DataProcess.pad_utterance(self.feature, self.sentence_sq)
            self.label = DataProcess.change_label_shape(self.train_y, self.n_labels)
        elif state is 'test':
            self.feature, self.sentence_sq = DataProcess.pad_sentence(self.test_x)
            self.feature, self.uttr_sq, self.sentence_sq = DataProcess.pad_utterance(self.feature, self.sentence_sq)
            self.label = DataProcess.change_label_shape(self.test_y, self.n_labels)
        else:
            self.feature, self.sentence_sq = DataProcess.pad_sentence(self.val_x)
            self.feature, self.uttr_sq, self.sentence_sq = DataProcess.pad_utterance(self.feature, self.sentence_sq)
            self.label = DataProcess.change_label_shape(self.val_y, self.n_labels)

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
    val_state = 1
    val_best = 0
    best_hamming = 1
    best_f1 = 0
    max_epoch = 500
    patience = 10
    lr_init = 0.001
    setup_seed(0)
    # path = '/dstc8/data/data.csv'
    path = '/data/cnn/dstc8.p'
    ld = MyData(os.getcwd() + path, state='train')
    if 1:
        with open(os.getcwd() + '/embedding/cnn_w2v.p', 'rb') as f:
            em_w = pickle.load(f)
    if 0:
        em_w = w2v.load_word2vec('glove', ld.word2index, 300)
        with open(os.getcwd() + '/embedding/cnn_w2v.p', 'wb') as f:
            pickle.dump(em_w, f)
    vb_size, emd_dim = em_w.shape
    print('embedding size:{}'.format(emd_dim))
    print('vocabulary size:{}'.format(vb_size))
    batch_size = BZ
    num_labels = 33
    model = CNN(em_w, vb_size, emd_dim, num_labels).to(device)

    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('total number of parameters: %d\n\n' % param_count)

    # 把 dataset 放入 DataLoader
    loader = data.DataLoader(
        dataset=ld,  # torch TensorDataset format
        batch_size=BZ,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
        drop_last=False,
        collate_fn=collate_func
    )
    val_set = MyData(os.getcwd() + path, state='val')
    val_loader = data.DataLoader(
        dataset=val_set,  # torch TensorDataset format
        batch_size=BZ,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
        drop_last=False,
        collate_fn=collate_func
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=0.00)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'max', factor=0.5,
                                                           verbose=True, patience=patience)

    num_iter_per_epoch = len(loader)
    # model.init_hidden_state(BZ)

    # writer = SummaryWriter("log/tensorboard", comment='DMLC')
    # writer.add_graph(model, (torch.randint(100, [BZ, 10, 50]).to(device),))
    start = time.time()
    # Training
    for epoch in range(max_epoch):
        train_loss = []
        train_acc = []
        train_hamming = []
        train_f1 = []
        model.train()
        for step, (batch_x, batch_y, sentence_sq, uttr_sq) in enumerate(loader):
            # 每一步 loader 释放一小批数据用来学习
            optimizer.zero_grad()
            input_batch = batch_x.squeeze(1).to(device)
            target_batch = batch_y.to(device)
            # model.init_hidden_state(BZ)
            output = model(input_batch)
            loss = criterion(output, target_batch)
            train_loss.append(loss)
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(target_batch.cpu().numpy(), output.cpu().detach().numpy(),
                                              list_metrics=["accuracy", "hamming", "micro-f1"])
            train_acc.append(training_metrics["accuracy"])
            train_hamming.append(training_metrics["hamming"])
            train_f1.append(training_metrics["micro-f1"])
            # print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {:.6f}, Accuracy: {:.4f}, Hamming: {:.6f}".format(
            #    epoch + 1,
            #    max_epoch,
            #    step + 1,
            #    num_iter_per_epoch,
            #    optimizer.param_groups[0]['lr'],
            #    loss, training_metrics["accuracy"],
            #    training_metrics["hamming"]))
            # writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + step)
            # writer.add_scalar('Train/hamming', training_metrics["hamming"], epoch * num_iter_per_epoch + step)
            # writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + step)
        print("Train Epoch: {}/{}, Lr: {}, Loss: {:.6f}, Accuracy: {:.4f}, Hamming: {:.6f}, micro-f1: {:.6f}".format(
            epoch + 1,
            max_epoch,
            optimizer.param_groups[0]['lr'],
            sum(train_loss) / num_iter_per_epoch,
            sum(train_acc) / num_iter_per_epoch,
            sum(train_hamming) / num_iter_per_epoch,
            sum(train_f1) / num_iter_per_epoch))
        if (epoch + 1) % patience == 0:
            end = time.time()
            print('cost time {:.4f}'.format(end - start))
            # # print('output: {}'.format(output))
            # print('epoch step:', '%04d' % (epoch + 1), '%04d' % (step + 1),
            #       'loss =', '{:.6f}'.format(loss), 'lr = {}'.format(optimizer.param_groups[0]['lr']))
            start = time.time()
        if val_state:
            model.eval()
            loss_ls = []
            val_label_ls = []
            val_pred_ls = []
            for step, (batch_x, batch_y, sentence_sq, uttr_sq) in enumerate(val_loader):
                num_sample = len(batch_y)
                input_batch = batch_x.squeeze(1).to(device)
                target_batch = batch_y.to(device)
                with torch.no_grad():
                    # model.init_hidden_state(num_sample)
                    val_predictions = model(input_batch)
                val_loss = criterion(val_predictions, target_batch)
                loss_ls.append(val_loss * num_sample)
                val_label_ls.extend(target_batch.clone().cpu().unsqueeze(0))
                val_pred_ls.append(val_predictions.clone().cpu())

            val_loss = sum(loss_ls) / val_set.__len__()
            val_pred = torch.cat(val_pred_ls, 0)
            val_label = torch.cat(val_label_ls, 0)
            val_metrics = get_evaluation(val_label.numpy(), val_pred.numpy(),
                                         list_metrics=["accuracy", "hamming", "micro-f1"])
            print("Val Epoch: {}/{}, Lr: {}, Loss: {:.6f}, Accuracy: {:.4f}, Hamming: {:.6f}, micro-f1: {:.6f}".format(
                epoch + 1,
                max_epoch,
                optimizer.param_groups[0]['lr'],
                val_loss, val_metrics["accuracy"],
                val_metrics["hamming"],
                val_metrics["micro-f1"]))
            scheduler.step(val_metrics["accuracy"])
            if val_metrics["accuracy"] > val_best:
                torch.save(model.state_dict(), os.getcwd() + '/best_model/cnn_best_acc_checkpoint.pt')
                val_best = val_metrics["accuracy"]
            if val_metrics['hamming'] < best_hamming:
                torch.save(model.state_dict(), os.getcwd() + '/best_model/cnn_best_hamming_checkpoint.pt')
                best_hamming = val_metrics['hamming']
            if val_metrics['micro-f1'] > best_f1:
                torch.save(model.state_dict(), os.getcwd() + '/best_model/cnn_best_micro_f1_checkpoint.pt')
                best_f1 = val_metrics['micro-f1']
            # val_best = max(val_best, val_metrics["accuracy"])
            # best_hamming = min(best_hamming, val_metrics['hamming'])
            # writer.add_scalar('Test/Loss', te_loss, epoch)
            # writer.add_scalar('Test/Hamming', test_metrics["hamming"], epoch)
            # writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)

        if optimizer.param_groups[0]['lr'] < 1e-4:
            break
    print("best val accuacy: {:.4f}".format(val_best))
    print("best val hamming loss: {:.6f}".format(best_hamming))


if __name__ == '__main__':
    main()
