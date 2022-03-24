#!/bin/sh -x
#PBS -l nodes=cgpu7:ppn=1
#PBS -N ...SIMCSE
#PBS -l walltime=30000:00:00       
#PBS -o /public/home/yuqi/lawliet/nlp/wentian/shell/log/cuda.out
#PBS -e /public/home/yuqi/lawliet/nlp/wentian/shell/log/cuda.err
#PBS -m abe

:> /public/home/yuqi/lawliet/nlp/wentian/shell/log/cuda.out
:> /public/home/yuqi/lawliet/nlp/wentian/shell/log/cuda.err

cd $PBS_O_WORKDIR
date
hostname
nvidia-smi
date