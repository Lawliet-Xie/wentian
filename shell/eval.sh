#!/bin/sh -x
#PBS -l nodes=cgpu5:ppn=5
#PBS -N ...SIMCSE
#PBS -l walltime=30000:00:00       
#PBS -o /public/home/yuqi/lawliet/nlp/wentian/shell/log/eval.out
#PBS -e /public/home/yuqi/lawliet/nlp/wentian/shell/log/eval.err
#PBS -m abe

:> /public/home/yuqi/lawliet/nlp/wentian/shell/log/eval.out
:> /public/home/yuqi/lawliet/nlp/wentian/shell/log/eval.err

cd $PBS_O_WORKDIR
date
hostname
nvidia-smi
/public/home/yuqi/anaconda3/bin/python3 /public/home/yuqi/lawliet/nlp/wentian/evaluate.py
date