import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def normal_pdf(pos, mu, sigma):
    a = pos - mu
    log_p = -1 * torch.mul(a, a) / (2 * sigma) - torch.log(sigma) / 2
    return F.softmax(log_p, dim=1)


class Gaussian_Position(nn.Module):
    def __init__(self, d_model, total_size, K=10):
        super(Gaussian_Position, self).__init__()
        # self.embedding = get_pe(d_model, K).to('cuda')
        # self.register_buffer('pe', self.embedding)
        self.embedding = nn.Parameter(torch.zeros([K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(total_size)], requires_grad=False).unsqueeze(1).repeat(1, K).to(
            'cuda')
        s = 0.0
        interval = total_size / K
        mu = []
        for _ in range(K):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(
            torch.tensor([torch.tensor([50.0], dtype=torch.float, requires_grad=True) for _ in range(K)]).unsqueeze(0))

    def forward(self, x):
        M = normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(M, self.embedding)
        # print(M)
        return x + pos_enc.unsqueeze(0).repeat(x.size(0), 1, 1)


class HARTransformer(nn.Module):
    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]):  # filters=[1, 5, 7]
        super(HARTransformer, self).__init__()
        # self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadedAttention(H, hidden_dim),
                         HAR_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        # x = self.pos_encoding(x)
        return self.model(x, mask)



class HAR_CNN(nn.Module):
    "Implements CNN equation."

    def __init__(self, d_model, d_ff, filters, dropout=0.1):
        super(HAR_CNN, self).__init__()
        self.kernel_num = int(d_ff)
        self.filter_sizes = filters
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d_model)
        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(
                                 in_channels=d_model,
                                 out_channels=self.kernel_num,
                                 kernel_size=filter_size,
                                 padding=int((filter_size - 1) / 2))
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))

    def forward(self, x):
        enc_outs = []
        for encoder in self.encoders:
            f_map = encoder(x.transpose(-1, -2))
            enc_ = f_map
            enc_ = F.relu(self.dropout(self.bn(enc_)))
            enc_outs.append(enc_.unsqueeze(dim=1))
        re = torch.div(torch.sum(torch.cat(enc_outs, 1), dim=1), 3)
        encoding = re

        return encoding.transpose(-1, -2)


class HARTrans(torch.nn.Module):
    def __init__(self, args):
        super(HARTrans, self).__init__()
        # self.softmax = torch.nn.LogSoftmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.transformer = HARTransformer(args.sequence_length, args.hlayers, args.hheads, 5)
        self.args = args
        self.kernel_num = 128
        self.kernel_num_v = 32
        self.filter_sizes = [5, 5]  # Reduced because of the input (5,50)
        self.filter_sizes_v = [50, 50]  # [2,2]
        self.pos_encoding_h = Gaussian_Position(args.sequence_length, args.dimensionality, args.K)
        self.pos_encoding_v = Gaussian_Position(args.dimensionality, args.sequence_length, args.K)
        if args.vlayers == 0:
            self.v_transformer = None
            self.dense = torch.nn.Linear(args.sequence_length, 2)
        else:
            self.v_transformer = HARTransformer(args.dimensionality, args.vlayers, args.vheads, 1)
            self.dense = torch.nn.Linear(self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v), args.output_dim)
        self.dense2 = torch.nn.Linear(self.kernel_num * len(self.filter_sizes), args.output_dim)
        self.dropout_rate = 0.5
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.encoders = []
        self.encoder_v = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=args.sequence_length,
                                             out_channels=self.kernel_num,
                                             kernel_size=filter_size).to('cuda')
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))
        for i, filter_size in enumerate(self.filter_sizes_v):
            enc_attr_name_v = "encoder_v_%d" % i
            self.__setattr__(enc_attr_name_v,
                             torch.nn.Conv1d(in_channels=5,
                                             out_channels=self.kernel_num_v,
                                             kernel_size=filter_size).to('cuda')
                             )
            self.encoder_v.append(self.__getattr__(enc_attr_name_v))

    def _aggregate(self, o, v=None):
        enc_outs = []
        enc_outs_v = []
        for encoder in self.encoders:
            f_map = encoder(o.transpose(-1, -2))
            enc_ = F.relu(f_map)
            k_h = enc_.size()[-1]
            enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            enc_ = enc_.squeeze(dim=-1)
            enc_outs.append(enc_)
        encoding = self.dropout(torch.cat(enc_outs, 1))
        q_re = F.relu(encoding)
        if self.v_transformer is not None:
            for encoder in self.encoder_v:
                f_map = encoder(v.transpose(-1, -2))
                enc_ = F.relu(f_map)
                k_h = enc_.size()[-1]
                enc_ = F.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs_v.append(enc_)
            encoding_v = self.dropout(torch.cat(enc_outs_v, 1))
            v_re = F.relu(encoding_v)
            q_re = torch.cat((q_re, v_re), dim=1)
        return q_re

    def forward(self, data):
        x = data.view(data.shape[0], 5, -1)
        x = self.pos_encoding_h(x)
        x = self.transformer(x)
        if self.v_transformer is not None:
            y = data.view(data.shape[0], 5, -1)
            y = y.transpose(-1, -2)
            y = self.pos_encoding_v(y)
            y = self.v_transformer(y)
            re = self._aggregate(x, y)
            predict = self.sigmoid(self.dense(re))
        else:
            re = self._aggregate(x)
            predict = self.sigmoid(self.dense2(re))
        return predict





