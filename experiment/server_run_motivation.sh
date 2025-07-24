#!/bin/bash

port=8000
for dataset in {"gsm8k",}
do
    for version in {"v2","v1"}
    do
        for model in {"qwen/qwen3-0.6b",}
        do
            # if [ $i -ge 8 ] && [ $version -eq "v1" ]; then
            #     continue
            # fi
            for i in {1,2,3,4,}
            do
                echo "Running Server with {model $model ,dataset=$dataset, USFL version=$version,client num =$i,LoRA=False,split_point=2}"
                python experiment/server_run.py -NC=${i} -V=${version} -SP=2 -M=${model} -P=${port}
                port=$((port + 1))
            done
        done
    done
done