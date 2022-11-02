import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class DMLC(nn.Module):
    def __init__(self, w2v_weight, vocab_size, embedding_dim, hidden_dim, batch_size, num_labels):
        super(DMLC, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.emd = EmbeddingLayer(w2v_weight, vocab_size, embedding_dim, hidden_dim, batch_size)
        self.sl = SentenceLayer(2 * hidden_dim, hidden_dim, batch_size, num_labels)

    def init_hidden_state(self, batch_size=None):
        self.emd.init_hidden_state(batch_size)
        self.sl.init_hidden_state(batch_size)

    def forward(self, input):
        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, w1 = self.emd(i.permute(1, 0))
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, w2 = self.sl(output)
        return output, w2


class EmbeddingLayer(nn.Module):
    def __init__(self, w2v_weight, vocab_size, embedding_dim, hidden_dim, batch_size, grad=True):
        super(EmbeddingLayer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        # self.embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(
            # torch.from_numpy(w2v_weight), padding_idx=0)
        # self.embedding.weight.requires_grad = False
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_dim)
        # self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_dim, 2 * hidden_dim))
        self.linear = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        nn.init.xavier_normal_(self.linear.weight)
        # self.word_bias = nn.Parameter(torch.zeros(1, 2 * hidden_dim))
        # self.linear.bias = nn.Parameter(torch.zeros(1, 2 * hidden_dim))
        self.content = nn.Linear(2 * hidden_dim, 1, bias=False)
        # self.content_weight = nn.Parameter(torch.Tensor(2 * hidden_dim, 1))
        nn.init.xavier_normal_(self.content.weight)

        self.hidden_state = None
        self.tanh = nn.Tanh()
        # self.drop = nn.Dropout(dropout)

    def init_hidden_state(self, batch_size=None):
        if batch_size:
            self.batch_size = batch_size
        self.hidden_state = torch.zeros(2, self.batch_size, self.hidden_dim).to(self.device)

    def forward(self, sequence):
        sequence_embedding = self.embedding(sequence)
        f_output, h_output = self.gru(sequence_embedding.float(), self.hidden_state)
        output = self.linear(f_output)
        output = self.tanh(output)
        output = self.content(output).squeeze(2).permute(1, 0)
        weight = F.softmax(output, dim=1)
        output = torch.bmm(f_output.permute(1, 0, 2).transpose(1, 2),
                           weight.unsqueeze(2)).squeeze(2).unsqueeze(0)
        return output, weight


class SentenceLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, num_labels):
        super(SentenceLayer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_dim)
        self.linear = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        nn.init.xavier_normal_(self.linear.weight)
        self.content = nn.Linear(2 * hidden_dim, 1, bias=False)
        nn.init.xavier_normal_(self.content.weight)
        self.hidden_state = None
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(2 * hidden_dim, num_labels)

    def init_hidden_state(self, batch_size=None):
        if batch_size:
            self.batch_size = batch_size
        self.hidden_state = torch.zeros(2, self.batch_size, self.hidden_dim).to(self.device)

    def forward(self, input):
        f_output, h_output = self.gru(input, self.hidden_state)
        output = self.linear(f_output)
        output = self.tanh(output)
        output = self.content(output).permute(1, 0, 2)
        weight = F.softmax(output, dim=1)
        output = torch.bmm(f_output.permute(1, 0, 2).transpose(1, 2),
                           weight).permute(0, 2, 1)
        output = self.fc(output).squeeze(1)
        return output, weight
