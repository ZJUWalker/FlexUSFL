#!/bin/bash

port=8000
for j in {1,}
do
    for dataset in {"gsm8k",}
    do
<<<<<<< HEAD
        for version in {"v1","v2"}
        do
            for model in {"qwen/qwen3-0.6b","meta-llama/llama3.2-1b"} #"meta-llama/llama3.2-1b"
=======
        for version in {"v2","v3"}
        do
            for model in {"qwen/qwen3-0.6b","qwen/qwen3-1.7b","meta-llama/llama3.2-1b"}
>>>>>>> 79d54401732e64f91d1d4bb0a01ba2d886ea1f91
            do
                # if [ $i -ge 8 ] && [ $version -eq "v1" ]; then
                #     continue
                # fi
<<<<<<< HEAD
                for i in {3,}
                do
                    echo "$j-th Running Server with {model $model ,dataset=$dataset, USFL version=$version,client num =$i,LoRA=False,split_point=2}"
                    python experiment/server_run.py -NC=${i} -V=${version} -SP=2 -M=${model} -P=${port} -CKPT -DS=${dataset}
=======
                for i in {32,}
                do
                    echo "$j-th Running Server with {model $model ,dataset=$dataset, USFL version=$version,client num =$i,LoRA=False,split_point=2}"
                    python experiment/server_run.py -NC=${i} -V=${version} -SP=2 -M=${model} -P=${port}
>>>>>>> 79d54401732e64f91d1d4bb0a01ba2d886ea1f91
                    port=$((port + 1))
                done
            done
        done
    done
done