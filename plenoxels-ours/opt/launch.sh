#!/bin/bash

echo Launching experiment $1
echo GPU $2
echo EXTRA ${@:3}

CKPT_DIR=ckpt/$1
mkdir -p $CKPT_DIR
NOHUP_FILE=$CKPT_DIR/log
echo CKPT $CKPT_DIR
echo LOGFILE $NOHUP_FILE

CUDA_VISIBLE_DEVICES=$2 python opt.py -t $CKPT_DIR ${@:3}
echo DETACH
