import math

import dgl.function as fn
import torch
import torch.nn as nn
from torch.nn import functional as F

class GraphSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 activation,
                 dropout,
                 bias=True,
                 use_pp=False,
                 use_lynorm=True):
        super(GraphSAGELayer, self).__init__()
        # The input feature size gets doubled as we concatenated the original
        # features with the new features.
        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        self.activation = activation
        self.use_pp = use_pp
        self._aggre_type = aggregator_type
        # aggregator type: mean/pool/lstm/gcn/attention
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_feats, in_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(in_feats, in_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(in_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=bias)

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
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)


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
            elif self._aggre_type == 'pool':
                g.ndata['h'] = F.relu(self.fc_pool(h))
                g.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h'))
                ah = g.ndata['h']
            elif self._aggre_type == 'lstm':
                g.ndata['h'] = h
                g.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
                ah = g.ndata['h']
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))
            # GraphSAGE GCN does not require fc_self.
            # if self._aggre_type == 'gcn':
            #     rst = self.fc_neigh(ah)
            # else:
            #     rst = self.fc_self(h) + self.fc_neigh(ah)

            h = self.concat(h, ah, norm)

        if self.dropout:
            h = self.dropout(h)

        #h = self.linear(h)


        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        ah = ah * norm
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
