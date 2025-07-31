#!/bin/bash

port=8000
max_seq_len=256
for j in {1,}
do
    for dataset in {"gsm8k",}
    do
        # if [ $dataset = "dialogsum" ]
        # then
        #     max_seq_len=512
        # else
        #     max_seq_len=128
        # fi
        for version in {"v1","v2"}
        do
            for model in {"qwen/qwen3-0.6b","meta-llama/llama3.2-1b"} #"meta-llama/llama3.2-1b"
            do
                # if [ $i -ge 8 ] && [ $version -eq "v1" ]; then
                #     continue
                # fi
                for i in {3,}
                do
                    echo "$j-th Running Clients with {model $model ,dataset $dataset, USFL version=$version, client num $i, max_seq_len=$max_seq_len, LoRA=True, split_point=2}"
                    python experiment/client_run.py -NC=${i} -V=${version} -L -SP=2 -M=${model} -P=${port} -B=1 -DS=${dataset} -SL=${max_seq_len}
                    port=$((port + 1))
                done
            done
        done
    done
done