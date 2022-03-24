#!/bin/sh -x
#PBS -l nodes=cgpu7:ppn=10
#PBS -N ...SIMCSE
#PBS -l walltime=30000:00:00       
#PBS -o /public/home/yuqi/lawliet/nlp/wentian/shell/log/train.out
#PBS -e /public/home/yuqi/lawliet/nlp/wentian/shell/log/train.err
#PBS -m abe

:> /public/home/yuqi/lawliet/nlp/wentian/shell/log/train.out
:> /public/home/yuqi/lawliet/nlp/wentian/shell/log/train.err

cd $PBS_O_WORKDIR
date
hostname
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /public/home/yuqi/anaconda3/bin/python3 /public/home/yuqi/lawliet/nlp/wentian/train_hard.py 
date