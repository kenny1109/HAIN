import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from rezero.transformer import RZTXEncoderLayer
from models_h_transformer.modules import PositionalEncoding


class DMLC(nn.Module):
    def __init__(self, w2v_weight, vocab_size, embedding_dim, hidden_dim, batch_size, num_labels):
        super(DMLC, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.emd = EmbeddingLayer(w2v_weight, vocab_size, embedding_dim, hidden_dim, batch_size)
        self.sl = SentenceLayer(embedding_dim, hidden_dim, batch_size, num_labels)
        self.dl = DecisionLayer(hidden_dim, num_labels)

    def init_hidden_state(self, batch_size=None):
        # self.emd.init_hidden_state(batch_size)
        self.sl.init_hidden_state(batch_size)

    def forward(self, input, w_mask, s_mask):
        output_list = []
        w1 = []
        input = input.permute(1, 0, 2)
        w_mask = w_mask.permute(1, 0, 2)
        for i, m in zip(input, w_mask):
            output, w = self.emd(i.permute(1, 0), m)
            output_list.append(output)
            w1.append(w)
        output = torch.cat(output_list, 0)
        output, w2 = self.sl(output, s_mask)
        output, w3 = self.dl(output)
        return output, [w1, w2, w3]


class EmbeddingLayer(nn.Module):
    def __init__(self, w2v_weight, vocab_size, embedding_dim, hidden_dim, batch_size, num_heads=6, head_dim=50):
        super(EmbeddingLayer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pos =  False
        self.embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(
            torch.from_numpy(w2v_weight), padding_idx=0)
        self.embedding.weight.requires_grad = False
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # nn.init.xavier_normal_(self.embedding.weight)

        if self.pos:
            self.pe = PositionalEncoding(d_model=embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, 4, 256, 0.1, activation='gelu')
        # encoder_layer = RZTXEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=256, dropout=0.1,
        #                                  activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)

        self.trans = nn.Linear(embedding_dim, head_dim * num_heads, bias=True)
        nn.init.xavier_normal_(self.trans.weight)
        self.query = nn.Parameter(torch.empty(head_dim * num_heads, 1))
        nn.init.xavier_normal_(self.query)
        self.linear = nn.Linear(head_dim * num_heads, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.linear.weight)

        # self.trans = nn.ModuleList()
        # self.contents = nn.ModuleList()
        # d = [50, 50, 50, 50]
        # # self.scale = np.sqrt(50)
        # self.scaled = d
        # for i in range(len(d)):
        #     l = nn.Linear(embedding_dim, d[i], bias=True)
        #     nn.init.xavier_normal_(l.weight)
        #     c = nn.Linear(d[i], 1, bias=False)
        #     nn.init.xavier_normal_(c.weight)
        #     self.trans.append(l)
        #     self.contents.append(c)
        # self.linear = nn.Linear(sum(d), embedding_dim, bias=True)
        # nn.init.xavier_normal_(self.linear.weight)

        self.drop = nn.Dropout(0.1)
        self.drop1 = nn.Dropout(0.1)
        self.tanh = nn.Tanh()

    def forward(self, sequence, mask):
        sequence_embedding = self.embedding(sequence)
        if self.pos:
            sequence_embedding = self.pe(sequence_embedding)
        else:
            sequence_embedding = self.drop(sequence_embedding)
        f_output = self.transformer_encoder(sequence_embedding, src_key_padding_mask=mask)

        output = self.trans(f_output)
        # check contiguous and make batch first
        output = output.contiguous().view(output.size()[0], output.size()[1],
                                          self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        query = self.query.contiguous().view(self.num_heads, self.head_dim).unsqueeze(0).permute(1, 2, 0)
        score = torch.matmul(output, query)
        mask = mask.unsqueeze(1).unsqueeze(3).expand(-1, self.num_heads, -1, -1)
        weight = F.softmax(score.masked_fill(mask, float('-inf')), dim=2)
        output = torch.matmul(output.transpose(2, 3), weight).squeeze(-1)
        output = output.contiguous().view(1, output.size()[0], -1)
        output = self.linear(output)
        output = self.drop1(output)

        # weight = []
        # o = []
        # for i in range(len(self.scaled)):
        #     output = self.trans[i](f_output)
        #     # output = self.drop(output)
        #     score = self.contents[i](output).squeeze(2).permute(1, 0)
        #     w = F.softmax(score.masked_fill(mask, float('-inf')), dim=1)
        #     # w = self.drop(w)
        #     output = torch.bmm(output.permute(1, 0, 2).transpose(1, 2),
        #                        w.unsqueeze(2)).squeeze(2).unsqueeze(0)
        #     weight.append(w.unsqueeze(1))
        #     o.append(output)
        # # print(torch.max(score))
        # output = torch.cat(o, dim=-1)
        # output = self.linear(output)
        # output = self.drop1(output)

        return output, weight


class SentenceLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, num_labels, num_heads=4, head_dim=50):
        super(SentenceLayer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pos = False

        if self.pos:
            self.pe = PositionalEncoding(d_model=input_dim)

        encoder_layer = nn.TransformerEncoderLayer(input_dim, 4, 512, 0.1, activation='gelu')
        # encoder_layer = RZTXEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=512, dropout=0.1,
        #                                  activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 1)

        self.trans = nn.Linear(hidden_dim, head_dim * num_heads, bias=True)
        nn.init.xavier_normal_(self.trans.weight)
        self.query = nn.Parameter(torch.empty(head_dim * num_heads, 1))
        nn.init.xavier_normal_(self.query)
        self.linear = nn.Linear(head_dim * num_heads, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.linear.weight)
        if 0:
            self.lb = nn.Linear(hidden_dim, hidden_dim * num_labels, bias=True)
            nn.init.xavier_normal_(self.lb.weight)

        # self.trans = nn.ModuleList()
        # self.contents = nn.ModuleList()
        # d = [50, 50, 50, 50]
        # self.scaled = d
        # for i in range(len(d)):
        #     l = nn.Linear(hidden_dim, d[i], bias=True)
        #     nn.init.xavier_normal_(l.weight)
        #     c = nn.Linear(d[i], 1, bias=False)
        #     nn.init.xavier_normal_(c.weight)
        #     self.trans.append(l)
        #     self.contents.append(c)
        # self.linear = nn.Linear(sum(d), hidden_dim, bias=True)
        # nn.init.xavier_normal_(self.linear.weight)
        # self.label = nn.ModuleList()
        # for i in range(num_labels):
        #     lb = nn.Linear(hidden_dim, hidden_dim, bias=True)
        #     nn.init.xavier_normal_(lb.weight)
        #     self.label.append(lb)

        self.drop = nn.Dropout(0.1)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.tanh = nn.Tanh()

    def forward(self, input, mask):
        if self.pos:
            input = self.pe(input)
        f_output = self.transformer_encoder(input, src_key_padding_mask=mask)

        output = self.trans(f_output)
        # check contiguous and make batch first
        output = output.contiguous().view(output.size()[0], output.size()[1],
                                          self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        query = self.query.contiguous().view(self.num_heads, self.head_dim).unsqueeze(0).permute(1, 2, 0)
        score = torch.matmul(output, query)
        mask = mask.unsqueeze(1).unsqueeze(3).expand(-1, self.num_heads, -1, -1)
        weight = F.softmax(score.masked_fill(mask, float('-inf')), dim=2)
        output = torch.matmul(output.transpose(2, 3), weight).squeeze(-1)

        output = output.contiguous().view(1, output.size()[0], -1)
        output = self.linear(output)
        output = self.drop(output)
        if 0:
            output = self.lb(output)
            output = self.drop2(output)
            output = output.contiguous().view(1, output.size()[1], self.num_labels, self.hidden_dim).squeeze(0).transpose(0, 1)

        # weight = []
        # o = []
        # for i in range(len(self.scaled)):
        #     output = self.trans[i](f_output)
        #     # output = self.drop(output)
        #     score = self.contents[i](output).squeeze(2).permute(1, 0)
        #     w = F.softmax(score.masked_fill(mask, float('-inf')), dim=1)
        #     # w = self.drop1(w)
        #     output = torch.bmm(output.permute(1, 0, 2).transpose(1, 2),
        #                        w.unsqueeze(2)).squeeze(2).unsqueeze(0)
        #     weight.append(w.unsqueeze(1))
        #     o.append(output)
        # output = torch.cat(o, dim=-1)
        # output = self.linear(output)
        # output = self.drop(output)
        # labels = []
        # for i in range(self.num_labels):
        #     l_out = self.label[i](output)
        #     # l_out = F.gelu(l_out)
        #     l_out = self.drop2(l_out)
        #     labels.append(l_out)
        # output = torch.cat(labels, dim=0)

        return output, weight


class DecisionLayer(nn.Module):
    def __init__(self, input_dim, num_class=33, hidden_dim=100, n_head=4, dropout=0.1):
        super(DecisionLayer, self).__init__()
        if 0:
            encoder_layer = nn.TransformerEncoderLayer(input_dim, 4, 512, 0.1, activation='gelu')
            # encoder_layer = RZTXEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=512, dropout=0.1,
            #                                  activation='gelu')
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 1)

            # self.out = nn.Linear(input_dim, 1)
            # nn.init.xavier_normal_(self.out.weight)

            self.topic_embedding = nn.Parameter(torch.empty(num_class, input_dim))
            self.mlp_1 = nn.Linear(num_class, hidden_dim)
            self.mlp_2 = nn.Linear(hidden_dim, 1)
            nn.init.xavier_normal_(self.topic_embedding)
            nn.init.xavier_normal_(self.mlp_1.weight)
            nn.init.xavier_normal_(self.mlp_2.weight)
            # self.ln = nn.LayerNorm(num_class)
            # self.scale = num_class ** -0.5

        if 1:
            self.out = nn.Linear(input_dim, num_class)

    def forward(self, input):
        attn_output_weights=None
        if 0:
            output = self.transformer_encoder(input)
            attn_output_weights=None
            #o, attn_output_weights = self.transformer_encoder.layers[0].self_attn(input, input, input)

            interaction = torch.matmul(self.topic_embedding, output.permute(1, 2, 0))
            # interaction = torch.matmul(output.permute(1, 0, 2), self.topic_embedding.transpose(1, 0))
            # interaction = self.ln(interaction)
            output = F.gelu(self.mlp_1(interaction))
            output = self.mlp_2(output).squeeze(-1)
        
        if 1:
            output = input
            output = self.out(output).squeeze()

        return  output, attn_output_weights
