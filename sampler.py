import os
import random

import dgl.function as fn
import torch
import torch.nn.functional as F
from partition_utils import *
import torch.nn as nn

class ClusterIter(object):
    '''The partition sampler given a DGLGraph and partition number.
    The metis is used as the graph partition backend.
    '''
    def __init__(self, dn, g, psize, batch_size, seed_nid, aggregator_type, in_feats, out_feats, cuda, gpu, use_pp=True):
        """Initialize the sampler.

        Paramters
        ---------
        dn : str
            The dataset name.
        g  : DGLGraph
            The full graph of dataset
        psize: int
            The partition number
        batch_size: int
            The number of partitions in one batch
        seed_nid: np.ndarray
            The training nodes ids, used to extract the training graph
        use_pp: bool
            Whether to use precompute of AX
        """
        
        self.use_pp = use_pp
        self.g = g.subgraph(seed_nid)
        self.g.copy_from_parent()
        self._in_feats = in_feats
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_feats, in_feats)
            if cuda:
                self.fc_pool = self.fc_pool.cuda()
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(in_feats, in_feats, batch_first=True)
            if cuda:
                self.lstm = self.lstm.cuda()
        self._aggre_type = aggregator_type
        # precalc the aggregated features from training graph only
        if use_pp:
            self.precalc(self.g)
            print('precalculating')

        self.psize = psize
        self.batch_size = batch_size
        # cache the partitions of known datasets&partition number
        if dn:
            fn = os.path.join('./datasets/', dn + '_{}.npy'.format(psize))
            if os.path.exists(fn):
                self.par_li = np.load(fn, allow_pickle=True)
            else:
                os.makedirs('./datasets/', exist_ok=True)
                self.par_li = get_partition_list(self.g, psize)
                np.save(fn, self.par_li)
        else:
            self.par_li = get_partition_list(self.g, psize)
        self.max = int((psize) // batch_size)
        random.shuffle(self.par_li)
        self.get_fn = get_subgraph

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
        return {'features': rst.squeeze(0)}

    def precalc(self, g):
        norm = self.get_norm(g)
        g.ndata['norm'] = norm
        with torch.no_grad():
            if self._aggre_type == 'mean':
                features = g.ndata['features']
                print("features shape, ", features.shape)
                g.update_all(fn.copy_src(src='features', out='m'),
                             fn.mean(msg='m', out='features'),
                             None)
                pre_feats = g.ndata['features']

            elif self._aggre_type == 'gcn':
                features = g.ndata['features']
                print("features shape, ", features.shape)
                g.update_all(fn.copy_src(src='features', out='m'),
                             fn.sum(msg='m', out='features'),
                             None)
                pre_feats = g.ndata['features'] * norm
                # use graphsage embedding aggregation style
            elif self._aggre_type == 'pool':
                features = F.relu(self.fc_pool(g.ndata['features']))
                print("features shape, ", features.shape)
                g.update_all(fn.copy_src(src='features', out='m'),
                             fn.max(msg='m', out='features'),
                             None)
                pre_feats = g.ndata['features']

            elif self._aggre_type == 'lstm':
                features = g.ndata['features']
                print("features shape, ", features.shape)
                g.update_all(fn.copy_src(src='features', out='m'),
                             self._lstm_reducer,
                             None)
                pre_feats = g.ndata['features']

            g.ndata['features'] = torch.cat([features, pre_feats], dim=1)

    # use one side normalization
    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.g.ndata['features'].device)
        return norm

    def __len__(self):
        return self.max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            result = self.get_fn(self.g, self.par_li, self.n,
                                 self.psize, self.batch_size)
            self.n += 1
            return result
        else:
            random.shuffle(self.par_li)
            raise StopIteration
