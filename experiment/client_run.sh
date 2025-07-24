#!/bin/bash

port=8000
for j in {1,}
do
    for dataset in {"gsm8k",}
    do
        for version in {"v2","v3"}
        do
            for model in {"qwen/qwen3-0.6b","qwen/qwen3-1.7b","meta-llama/llama3.2-1b"}
            do
                # if [ $i -ge 8 ] && [ $version -eq "v1" ]; then
                #     continue
                # fi
                for i in {32,}
                do
                    echo "$j-th Running Clients with {model $model ,dataset $dataset, USFL version=$version, client num $i, LoRA=True,split_point=2}"
                    python experiment/client_run.py -NC=${i} -V=${version} -L -SP=2 -M=${model} -P=${port} -B=1 -DS=${dataset}
                    port=$((port + 1))
                done
            done
        done
    done
done