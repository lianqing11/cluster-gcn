#!/bin/bash
aggregators=("gcn" "mean" "pool" "lstm")
for aggregator_type in ${aggregators[@]}
#aggregator_type=gcn
do
    for n_layer in {1..6}
    do
        log_path=$aggregator_type-ly$n_layer-self-loop-ppi-non-sym-pp-cluster-2-2-wd-0
        echo $aggregator_type
        echo $log_path
        python cluster_gcn.py --gpu 0 --dataset ppi --lr 1e-2 --weight-decay 0.0 --psize 50 --batch-size 1 --n-epochs 300 \
          --n-hidden 2048 --n-layers $n_layer --log-every 100 --use-pp --self-loop \
          --note $log_path --dropout 0.2 --use-val --normalize --aggregator_type $aggregator_type
    done
done