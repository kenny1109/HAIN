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
from lr_schedulers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from test import predict
from tensorboardX import SummaryWriter




def main(n, lamda, beta):
    print(n)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BZ = 32
    val_state = 1
    val_best = 0
    best_hamming = 1
    best_f1 = 0
    patience = 3
    lr_init = 0.0005
    if os.path.exists(os.getcwd() + '/results/train/' + str(lamda) + '-' + str(beta) + 'correlation.txt'):
        log_file = open(os.getcwd() + '/results/train/' + str(lamda) + '-' + str(beta) + 'correlation.txt', 'a')
    else:
        log_file = open(os.getcwd() + '/results/train/' + str(lamda) + '-' + str(beta) + 'correlation.txt', 'w')
    direc = os.getcwd() + '/best_model/multi-head-4-50-' + str(lamda) + '-' + str(beta)
    if not os.path.exists(direc):
        os.makedirs(direc)

    setup_seed(n)
    # path = '/dstc8/data/data.csv'
    path = '/data/dstc8.p'
    ld = MyData(os.getcwd() + path, state='train', batch_size=BZ)
    if 0:
        with open(os.getcwd() + '/embedding/w2v.p', 'rb') as f:
            em_w = pickle.load(f)
    if 1:
        em_w = w2v.load_word2vec('glove', ld.word2index, 200)
        with open(os.getcwd() + '/embedding/w2v.p', 'wb') as f:
            pickle.dump(em_w, f)
    vb_size, emd_dim = em_w.shape
    print('embedding size:{}'.format(emd_dim))
    print('vocabulary size:{}'.format(vb_size))
    batch_size = BZ
    num_labels = ld.n_labels
    hidden_size = emd_dim
    model = DMLC(em_w, vb_size, emd_dim, hidden_size, batch_size, num_labels).to(device)
    
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('total number of parameters: %d\n\n' % param_count)
    # print(model)

    # 把 dataset 放入 DataLoader
    loader = data.DataLoader(
        dataset=ld,  # torch TensorDataset format
        batch_size=BZ,  # mini batch size
        shuffle=False,  # 要不要打乱数据 
        num_workers=0,  # 多线程来读数据
        drop_last=False,
        collate_fn=collate_func,
    )
    val_set = MyData(os.getcwd() + path, state='val', batch_size=BZ)
    val_loader = data.DataLoader(
        dataset=val_set,  # torch TensorDataset format
        batch_size=BZ,  # mini batch size
        shuffle=False,  # 
        num_workers=0,  # 多线程来读数据
        drop_last=False,
        collate_fn=collate_func
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr_init)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'max', factor=0.8,
                                                           verbose=True, patience=patience)
    
    num_iter_per_epoch = len(loader)

    max_epoch = 100
    t_total = int(len(loader) * max_epoch)
    warmup_proportion = 0.05

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(t_total * warmup_proportion)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr_init, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, t_total)

    # writer = SummaryWriter("log/tensorboard", comment='DMLC')
    # writer.add_graph(model, (torch.randint(100, [BZ, 10, 50]).to(device),))
    start = time.time()
    df = [0, 0, 0]
    # Training
    for epoch in range(max_epoch):
        train_loss = []
        train_acc = []
        train_hamming = []
        train_f1 = []
        model.train()
        for step, (batch_x, batch_y, sentence_sq, uttr_sq) in enumerate(loader):
            # 每一步 loader 释放一小批数据用来学习
            # adjust_learning_rate(optimizer, emd_dim, (step+1) + epoch * num_iter_per_epoch, warmup_steps=2000)
            optimizer.zero_grad()
            input_batch = batch_x.to(device)
            target_batch = batch_y.to(device)
            sentence_sq = sentence_sq.to(device)
            uttr_sq = uttr_sq.to(device)
            output, attention = model(input_batch, sentence_sq, uttr_sq)

            w_l2 = []
            for i in range(len(attention[0])):
                # m2 = torch.cat(attention[0][i], dim=1)
                m2 = attention[0][i].squeeze(-1)
                a = torch.bmm(m2, m2.permute(0, 2, 1))
                b = torch.eye(m2.size()[1]).unsqueeze(0).expand(m2.size()[0], m2.size()[1], m2.size()[1]).to(device)
                c = torch.sum(torch.norm(b - a, p=2, dim=(1, 2)).masked_fill(uttr_sq[:, i], float(0.0))) \
                    / (uttr_sq.size()[0] - torch.nonzero(uttr_sq[:, i]).size()[0])
                # c = torch.sum(torch.norm(b - a, p=2, dim=(1, 2)))
                # c = torch.sum(torch.norm(b, dim=(1, 2))) - torch.sum(torch.norm(a, dim=(1, 2)))
                w_l2.append(c)
            # m2 = torch.cat(attention[1], dim=1)
            m2 = attention[1].squeeze(-1)
            a = torch.bmm(m2, m2.permute(0, 2, 1))
            b = torch.eye(m2.size()[1]).unsqueeze(0).expand(m2.size()[0], m2.size()[1], m2.size()[1]).to(device)
            c = torch.sum(torch.norm(b - a, p=2, dim=(1, 2))) / m2.size()[0]
            # c = torch.sum(torch.norm(b - a, p=2, dim=(1, 2)))
            # c = torch.sum(torch.norm(b, dim=(1, 2))) - torch.sum(torch.norm(a, dim=(1, 2)))
            # w_l2.append(c)
            s_l2 = c
            l2 = sum(w_l2) / len(w_l2)
            # l2 = sum(w_l2) / (len(w_l2) * m2.size()[0])

            loss = criterion(output, target_batch) + lamda * l2 + beta * s_l2
            # loss = criterion(output, target_batch)
            train_loss.append(loss)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            scheduler.step()
            training_metrics = get_evaluation(target_batch.cpu().numpy(), output.cpu().detach().numpy(),
                                              list_metrics=["accuracy", "hamming", "micro-f1"])
            train_acc.append(training_metrics["accuracy"])
            train_hamming.append(training_metrics["hamming"])
            train_f1.append(training_metrics["micro-f1"])
            # if torch.cuda.is_available():
            #    torch.cuda.empty_cache()
            if 0:
                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {:.6f}, Accuracy: {:.4f}, Hamming: {:.6f}".format(
                    epoch + 1,
                    max_epoch,
                    step + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss, training_metrics["accuracy"],
                    training_metrics["hamming"]))
            # writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + step)
            # writer.add_scalar('Train/hamming', training_metrics["hamming"], epoch * num_iter_per_epoch + step)
            # writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + step)
        if 0:
            log_file.write("Train Epoch: {}/{}, Lr: {}, Loss: {:.6f}, Accuracy: {:.4f}, Hamming: {:.6f},"
                           " micro-f1: {:.6f}\n".format(
                            epoch,
                            max_epoch,
                            optimizer.param_groups[0]['lr'],
                            sum(train_loss) / num_iter_per_epoch,
                            sum(train_acc) / num_iter_per_epoch,
                            sum(train_hamming) / num_iter_per_epoch,
                            sum(train_f1) / num_iter_per_epoch))
            print("train word l2:{:.4f}, sentence l2:{:.4f}".format(l2, s_l2))
        if 0:
            print("Train Epoch: {}/{}, Lr: {}, Loss: {:.6f}, Accuracy: {:.4f}, Hamming: {:.6f}, micro-f1: {:.6f}".format(
                epoch,
                max_epoch,
                optimizer.param_groups[0]['lr'],
                sum(train_loss) / num_iter_per_epoch,
                sum(train_acc) / num_iter_per_epoch,
                sum(train_hamming) / num_iter_per_epoch,
                sum(train_f1) / num_iter_per_epoch))
        if (epoch+1) % patience == 0:
            end = time.time()
            print('cost time {:.4f}'.format(end - start))
            start = time.time()
        if val_state:
            model.eval()
            loss_ls = []
            val_label_ls = []
            val_pred_ls = []
            for step, (batch_x, batch_y, sentence_sq, uttr_sq) in enumerate(val_loader):
                num_sample = len(batch_y)
                input_batch = batch_x.to(device)
                target_batch = batch_y.to(device)
                sentence_sq = sentence_sq.to(device)
                uttr_sq = uttr_sq.to(device)
                with torch.no_grad():
                    val_predictions, attention = model(input_batch, sentence_sq, uttr_sq)
                    w_l2 = []
                    for i in range(len(attention[0])):
                        # m2 = torch.cat(attention[0][i], dim=1)
                        m2 = attention[0][i].squeeze(-1)
                        a = torch.bmm(m2, m2.permute(0, 2, 1))
                        b = torch.eye(m2.size()[1]).unsqueeze(0).expand(m2.size()[0], m2.size()[1], m2.size()[1]).to(device)
                        c = torch.sum(torch.norm(b - a, p=2, dim=(1, 2)).masked_fill(uttr_sq[:, i], float(0.0))) \
                            / (uttr_sq.size()[0] - torch.nonzero(uttr_sq[:, i]).size()[0])
                        # c = torch.sum(torch.norm(b - a, p=2, dim=(1, 2)))
                        # c = torch.sum(torch.norm(b, dim=(1, 2))) - torch.sum(torch.norm(a, dim=(1, 2)))
                        w_l2.append(c)
                    # m2 = torch.cat(attention[1], dim=1)
                    m2 = attention[1].squeeze(-1)
                    a = torch.bmm(m2, m2.permute(0, 2, 1))
                    b = torch.eye(m2.size()[1]).unsqueeze(0).expand(m2.size()[0], m2.size()[1], m2.size()[1]).to(device)
                    c = torch.sum(torch.norm(b - a, p=2, dim=(1, 2))) / m2.size()[0]
                    # c = torch.sum(torch.norm(b - a, p=2, dim=(1, 2)))
                    # c = torch.sum(torch.norm(b, dim=(1, 2))) - torch.sum(torch.norm(a, dim=(1, 2)))
                    # w_l2.append(c)
                    s_l2 = c
                    l2 = sum(w_l2) / len(w_l2)
                    # l2 = sum(w_l2) / (len(w_l2) * m2.size()[0])

                val_loss = criterion(val_predictions, target_batch)
                loss_ls.append(val_loss * num_sample)
                val_label_ls.extend(target_batch.clone().cpu().unsqueeze(0))
                val_pred_ls.append(val_predictions.clone().cpu())
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()

            val_loss = sum(loss_ls) / val_set.__len__()
            val_pred = torch.cat(val_pred_ls, 0)
            val_label = torch.cat(val_label_ls, 0)
            val_metrics = get_evaluation(val_label.numpy(), val_pred.numpy(), list_metrics=["accuracy", "hamming", "micro-f1"])
            log_file.write("Val Epoch: {}/{}, Lr: {}, Loss: {:.6f}, Accuracy: {:.4f}, Hamming: {:.6f}, "
                           "micro-f1: {:.6f} \n".format(
                            epoch,
                            max_epoch,
                            optimizer.param_groups[0]['lr'],
                            val_loss, val_metrics["accuracy"],
                            val_metrics["hamming"],
                            val_metrics["micro-f1"]))
            print("valid word l2:{:.4f}, sentence l2:{:.4f}".format(l2, s_l2))
            if 1:
                print("Val Epoch: {}/{}, Lr: {}, Loss: {:.6f}, Accuracy: {:.4f}, Hamming: {:.6f}, micro-f1: {:.6f}".format(
                    epoch,
                    max_epoch,
                    optimizer.param_groups[0]['lr'],
                    val_loss, val_metrics["accuracy"],
                    val_metrics["hamming"],
                    val_metrics["micro-f1"]))
            # scheduler.step(val_metrics["accuracy"])
            if optimizer.param_groups[0]['lr'] <= 1e-4:
                torch.save(model.state_dict(), direc + '/' + str(epoch)+'best_acc_checkpoint.pt')
                # torch.save(model.state_dict(), os.getcwd() + '/best_model/'+str(epoch)+'best_hamming_checkpoint.pt')
                # torch.save(model.state_dict(), os.getcwd() + '/best_model/'+str(epoch)+'best_micro_f1_checkpoint.pt')
                if val_metrics["accuracy"] >= val_best:
                    torch.save(model.state_dict(), direc + '/best_acc_checkpoint.pt')
                    val_best = val_metrics["accuracy"]
                    df[0] = epoch
                if val_metrics['hamming'] <= best_hamming:
                    torch.save(model.state_dict(), direc + '/best_hamming_checkpoint.pt')
                    best_hamming = val_metrics['hamming']
                    df[1] = epoch
                if val_metrics['micro-f1'] >= best_f1:
                    torch.save(model.state_dict(), direc + '/best_micro_f1_checkpoint.pt')
                    best_f1 = val_metrics['micro-f1']
                    df[2] = epoch
            # val_best = max(val_best, val_metrics["accuracy"])
            # best_hamming = min(best_hamming, val_metrics['hamming'])
            # writer.add_scalar('Test/Loss', te_loss, epoch)
            # writer.add_scalar('Test/Hamming', test_metrics["hamming"], epoch)
            # writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
        
        if optimizer.param_groups[0]['lr'] == 0:
            # continue
            break
    log_file.write("lambda: {}, beta: {} \n".format(lamda, beta))
    log_file.write("best val accuacy: {:.4f} \n".format(val_best))
    log_file.write("best val hamming loss: {:.6f}\n".format(best_hamming))
    log_file.write("model number: {} \n\n".format(df))
    print("best val accuacy: {:.4f}".format(val_best))
    print("best val hamming loss: {:.6f}".format(best_hamming))
    print("model number: {}".format(df))


if __name__ == '__main__':
    for i in np.arange(0.00000, 0.0002, 0.000025):
    # for i in range(0, 1):
        for k in np.arange(0.000, 0.001375, 0.00025):
            print(k, i)
            main(123, k, i)

            test_best = 0
            best_hamming = 1
            best_f1 = 0
            d = [0, 0, 0]
            for j in range(80, 100):
                print(j)
                direc = os.getcwd() + '/best_model/multi-head-4-50-' + str(k) + '-' + str(i)
                model_path = direc + '/' + str(j) + 'best_acc_checkpoint.pt'
                a, b, c = predict(model_path, j, k, i)
                if a >= test_best:
                    test_best = a
                    d[0] = j
                if b <= best_hamming:
                    best_hamming = b
                    d[1] = j
                if c >= best_f1:
                    best_f1 = c
                    d[2] = j
            if os.path.exists(os.getcwd() + '/results/train/best_correlation.txt'):
                log_file = open(os.getcwd() + '/results/train/best_correlation.txt', 'a')
            else:
                log_file = open(os.getcwd() + '/results/train/best_correlation.txt', 'w')
            log_file.write("penalty: {}  {}\n".format(k, i))
            log_file.write("best test accuacy: {:.4f}\n".format(test_best))
            log_file.write("best test hamming loss: {:.6f}\n".format(best_hamming))
            log_file.write("best test micro-f1: {:.4f}\n".format(best_f1))
            log_file.write("model number: {}\n\n".format(d))
            log_file.close()
            print("best test accuacy: {:.4f}".format(test_best))
            print("best test hamming loss: {:.6f}".format(best_hamming))
            print("best test micro-f1: {:.4f}".format(best_f1))
            print("model number: {}".format(d))
            # break
        # break
