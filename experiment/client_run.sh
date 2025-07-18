#!/bin/bash

port=8000
for version in {"v1","v2"}
do
    for i in {1,2,4,6,8,16}
    do
        for model in {"meta-llama/llama3.2-1b",}
        do
            echo "Running Clients with {model $model ,USFL version=$version, client num $i, LoRA=True,split_point=2}"
            python experiment/client_run.py -NC=${i} -V=${version} -L -SP=2 -M=${model} -P=${port}
            port=$((port + 1))
        done
    done
done