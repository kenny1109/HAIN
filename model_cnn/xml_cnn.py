import torch
import torch.nn as nn
import torch.nn.functional as F


class XMLCNN(nn.Module):
    def __init__(self, w2v_weight, vocab_size, embedding_dim, hidden_dim, output_size, pool_units=32, sequence_len=600,
                 num_filters=128, kernel_sizes=[2, 4, 8], drop_prob=0.5):
        super(XMLCNN, self).__init__()

        self.num_filters = num_filters
        self.pool_units = pool_units
        self.embedding_dim = embedding_dim
        # 1. embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(
            torch.from_numpy(w2v_weight), padding_idx=0)
        self.embedding.weight.requires_grad = False
        # 2. convolutional layers
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), stride=(2, embedding_dim))
            for k in kernel_sizes])
        self.pool = nn.ModuleList()
        # 3. final, fully-connected layer for classification
        pool_tpye = 'max'
        fin_l_out_size = 0
        if pool_tpye == 'average':
            for ks in kernel_sizes:
                l_out_size = self.out_size(sequence_len, ks, stride=2)
                pool_size = l_out_size // self.pool_units
                l_pool = nn.AvgPool1d(pool_size, stride=None, count_include_pad=True)
                pool_out_size = (int((l_out_size - pool_size)/pool_size) + 1) * num_filters
                fin_l_out_size += pool_out_size
                self.pool.append(l_pool)

        elif pool_tpye == 'max':
            for ks in kernel_sizes:
                l_out_size = self.out_size(sequence_len, ks, stride=2)
                pool_size = 128
                stride = 64
                l_pool = nn.MaxPool1d(pool_size, stride=stride)
                pool_out_size = (int((l_out_size - pool_size) / stride) + 1) * num_filters
                fin_l_out_size += pool_out_size
                self.pool.append(l_pool)
        print(fin_l_out_size)

        self.fin_layer = nn.Linear(fin_l_out_size, hidden_dim)
        nn.init.xavier_normal_(self.fin_layer.weight)
        self.fc = nn.Linear(hidden_dim, output_size)
        nn.init.xavier_normal_(self.fc.weight)
        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(0.1)

    @staticmethod
    def out_size(l_in, kernel_size, padding=0, dilation=1, stride=1):
        a = l_in + 2 * padding - dilation * (kernel_size - 1) - 1
        b = int(a / stride)
        return b + 1

    def conv_and_pool(self, x, conv, pool):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 200
        # x = F.relu(conv(x)).squeeze(3)
        x = conv(x).squeeze(3)
        x = F.relu(x)

        # 1D pool over conv_seq_length
        # squeeze to get size: (batch_size, num_filters)
        # x = F.max_pool1d(x, x.size(2) // self.pool_unit)
        x = pool(x)
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, x):
        """
        Defines how a batch of inputs, x, passes through the model layers.
        Returns a single, sigmoid-activated class score as output.
        """
        # embedded vectors
        embeds = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embeds = self.dropout2(embeds)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)

        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv, pool) for conv, pool in zip(self.convs_1d, self.pool)]

        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.fin_layer(x)
        x = F.relu(x)
        x = self.dropout(x)

        # final logit
        x = self.fc(x)

        # sigmoid-activated --> a class score
        return x

