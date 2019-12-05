#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
for hs in 12
do
for es in 64,64
do
for snm in 1.0 2.0
do
for spr in 0.1 0.3 0.5
do
for lr in 0.0003
do
for bs in 1024
do
echo "Hyper parameters: " $hs $es $snm $lr $bs $spr
  python main.py -k 10 --hash_size $hs --encoder_structure $es --hashing_type MultivariateBernoulli --distance_type L2 --data_id glove_100 --logger_type cometml --learner_type siamese --siamese_positive_margin 0.0 --siamese_negative_margin $snm --siamese_positive_rate $spr --batch_size $bs --learning_rate $lr --log_tags query_size
done
done
done
done
done
done
