import math
from dgl.nn.pytorch import edge_softmax

import dgl.function as fn
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch as th

class GraphSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 activation,
                 dropout,
                 num_heads=1,
                 bias=True,
                 use_pp=False,
                 use_lynorm=True):
        super(GraphSAGELayer, self).__init__()
        # The input feature size gets doubled as we concatenated the original
        # features with the new features.
        self._in_feats = in_feats
        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        self.num_heads = num_heads,
        self.activation = activation
        self.use_pp = use_pp
        self._aggre_type = aggregator_type
        # aggregator type: mean/pool/lstm/gcn/attention
        self.leaky_relu = nn.LeakyReLU(0.2)
        if self.use_pp is True:
            if aggregator_type == 'pool':
                self.fc_pool = nn.Linear(in_feats, in_feats)
            elif aggregator_type == 'lstm':
                self.lstm = nn.LSTM(in_feats, in_feats, batch_first=True)
            elif aggregator_type == 'attn':
                self.fc_attn = nn.Linear(in_feats, in_feats*self.num_heads)
                self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, in_feats)))
                self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, in_feats)))



        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.linear.weight.size(1))
        # self.linear.weight.data.uniform_(-stdv, stdv)
        # if self.linear.bias is not None:
        #     self.linear.bias.data.uniform_(-stdv, stdv)
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self.use_pp is True:
            if self._aggre_type == 'pool':
                nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
            if self._aggre_type == 'lstm':
                self.lstm.reset_parameters()


    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_feats)),
             m.new_zeros((1, batch_size, self._in_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'h': rst.squeeze(0)}

    def forward(self, g, h):
        g = g.local_var()
        if not self.use_pp or not self.training:
            norm = self.get_norm(g)

            # g.ndata['h'] = h
            # g.update_all(fn.copy_src(src='h', out='m'),
            #              fn.sum(msg='m', out='h'))
            # ah = g.ndata.pop('h')

            if self._aggre_type == 'mean':
                g.ndata['h'] = h
                g.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'h'))
                ah = g.ndata.pop('h')
            elif self._aggre_type == 'gcn':
                g.ndata['h'] = h
                g.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h'))
                # divide in_degrees
                # degs = graph.in_degrees().float()
                # degs = degs.to(feat.device)
                # h_neigh = (graph.ndata['neigh'] + graph.ndata['h']) / (degs.unsqueeze(-1) + 1)
                ah = g.ndata.pop('h')
                ah = ah * norm
            elif self._aggre_type == 'pool':
                g.ndata['h'] = F.relu(self.fc_pool(h))
                g.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'))
                ah = g.ndata['h']
            elif self._aggre_type == 'lstm':
                g.ndata['h'] = h
                g.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
                ah = g.ndata['h']
            elif self._aggre_type == 'attn':
                feat = self.fc(h).view(-1, self.num_heads, self.in_feats)
                el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
                g.ndata.update({'ft': feat, 'el': el, 'er': er})
                g.apply_edges(fn.u_add_v('el', 'er', 'e'))
                e = self.leaky_relu(g.edata.pop('e'))
                g.edata['a'] = edge_softmax(g, e)
                g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
                ah = g.ndata['ft']

            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            h = self.concat(h, ah, norm)
        if self.dropout:
            h = self.dropout(h)
        # GraphSAGE GCN does not require fc_self.
        # if self._aggre_type == 'gcn':
        #     rst = self.fc_neigh(ah)
        # else:
        #     rst = self.fc_self(h) + self.fc_neigh(ah)
        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        h = torch.cat((h, ah), dim=1)
        return h

    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.linear.weight.device)
        return norm

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 use_pp,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphSAGELayer(in_feats, n_hidden, aggregator_type, activation=activation,
                                        dropout=dropout, use_pp=use_pp, use_lynorm=True))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphSAGELayer(n_hidden, n_hidden, aggregator_type, activation=activation, dropout=dropout,
                             use_pp=False, use_lynorm=True))
        # output layer
        self.layers.append(GraphSAGELayer(n_hidden, n_classes, aggregator_type, activation=None,
                                        dropout=dropout, use_pp=False, use_lynorm=False))

    def forward(self, g):
        h = g.ndata['features']
        for layer in self.layers:
            h = layer(g, h)
        return h
