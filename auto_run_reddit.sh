#!/bin/bash
aggregators=("gcn" "mean" "pool" "lstm")
for aggregator_type in ${aggregators[@]}
#aggregator_type=gcn
do
    for n_layer in {1..6}
    do
        log_path=$aggregator_type-ly$n_layer-self-loop-reddit-non-sym-pp-cluster-2-2-wd-0
        echo $aggregator_type
        echo $log_path
        python cluster_gcn.py --gpu 0 --dataset reddit-self-loop --lr 1e-2 --weight-decay 0.0 --psize 1500 --batch-size 20 \
          --n-epochs 30 --n-hidden 128 --n-layers $n_layer --log-every 100 --use-pp --self-loop \
          --note $log_path --dropout 0.2 --use-val --normalize --aggregator_type $aggregator_type
    done
done

