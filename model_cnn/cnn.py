import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, w2v_weight, vocab_size, embedding_dim, output_size, squence_len=600,
                 num_filters=200, kernel_sizes=[3,4,5], drop_prob=0.5):
        super(CNN, self).__init__()

        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        # 1. embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(
            torch.from_numpy(w2v_weight), padding_idx=0)
        self.embedding.weight.requires_grad = False
        # 2. convolutional layers
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), stride=(2, embedding_dim))
            for k in kernel_sizes])
        # 3. final, fully-connected layer for classification
        self.fin_layer = nn.Linear(len(kernel_sizes) * num_filters, len(kernel_sizes) * num_filters)
        nn.init.xavier_normal_(self.fin_layer.weight)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size)
        nn.init.xavier_normal_(self.fc.weight)
        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(0.1)

    def conv_and_pool(self, x, conv):
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
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        """
        Defines how a batch of inputs, x, passes through the model layers.
        Returns a single, sigmoid-activated class score as output.
        """
        # embedded vectors
        embeds = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        # embeds = self.dropout2(embeds)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)

        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]

        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.fin_layer(x)
        x = F.relu(x)
        x = self.dropout(x)

        # final logit
        x = self.fc(x)

        # sigmoid-activated --> a class score
        return x

