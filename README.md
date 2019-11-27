
# Comp 5331 course project.

------------
Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
Dependencies
------------
- Python 3.7+(for string formatting features)
- PyTorch 1.1.0+
- metis
- sklearn


* install clustering toolkit: metis and its Python interface.

  download and install metis: http://glaros.dtc.umn.edu/gkhome/metis/metis/download

  METIS - Serial Graph Partitioning and Fill-reducing Matrix Ordering ([official website](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview))

```
1) Download metis-5.1.0.tar.gz from http://glaros.dtc.umn.edu/gkhome/metis/metis/download and unpack it
2) cd metis-5.1.0
3) make config shared=1 prefix=~/.local/
4) make install
5) export METIS_DLL=~/.local/lib/libmetis.so
6) `pip install metis`
```

quick test to see whether you install metis correctly:

```
>>> import networkx as nx
>>> import metis
>>> G = metis.example_networkx()
>>> (edgecuts, parts) = metis.part_graph(G, 3)
```


### Run experiments
For ppi dataset with gcn, mean, pooling aggregator:
```
sh auto_gcn_mean_pool_run_ppi.sh
```
For ppi dataset with lstm aggregator:
```
sh auto_lstm_run_ppi.sh
```

For ppi dataset with attention aggregator:
```
sh auto_attn_run_ppi.sh 
```

-------------------

For reddit dataset with gcn, mean, pooling aggregator:
```
sh auto_gcn_mean_pool_run_reddit.sh
```


For reddit dataset with attention aggregator:
```
sh auto_attn_run_reddit.sh 
```

You can get tune the hyper-param in the script and get the result from the "log" folder.

### Acknowledgements
This code is heavily borrowed from [DGL's implementation](https://github.com/dmlc/dgl/tree/master/examples/pytorch/cluster_gcn)
