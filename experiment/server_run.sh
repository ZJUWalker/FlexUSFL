#!/bin/bash

port=8000
for i in {1,2,4,6,8}
do
    for model in {"qwen/qwen3-0.6b","qwen/qwen3-1.7b","meta-llama/llama3.2-1b"}
    do
        echo "Running Server with {client num $i, model $model ,USFL version=V1,LoRA=False,split_point=2}"
        python experiment/server_run.py -NC=${i} -V=v1 -SP=2 -M=${model} -P=${port}
        port=$((port + 1))
    done
done