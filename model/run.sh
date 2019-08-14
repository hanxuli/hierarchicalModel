#! /bin/bash

# 十分之一数据
train_path=../data/IMDB/train.txt
test_path=../data/IMDB/test.txt
val_path=../data/IMDB/dev.txt
dataName=IMDB

for units in 32 64 128 256 512; do
for batch_size in  512  256 128 64 32; do
for epochs in 10: do
for num_classes in 10: do
for optimizer in SGD Adam Adadelta Adagrad RMSprop; do
for lr in 0.00001 0.0001  0.001 0.01; do
for  dropout in 0.1 0.2 0.3 0.4 0.5: do
for decay in 0.0001 0.00001 0.001: do
for maxlen in 2000:do
for   Name in LocalAttention_indRnn:do


python LocalAttention.py --dataName $dataName --train $train_path --test $test_path --Name $Name\
     --val $val_path  --epochs  $epochs --units $units --batch_size $batch_size  --maxlen $maxlen\
     --num_classes $num_classes --lr $lr  --optimizer $optimizer --dropout $dropout --decay $decay


done
done
done
done
done
done
done
done
done
done
done
done
done
done


