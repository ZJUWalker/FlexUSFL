#!/bin/bash

port=8011

for j in {1,}
do
    for dataset in {"gsm8k",}
    do
        for version in {"v1",}
        do
            for model in {"meta-llama/llama3.2-1b",} #"meta-llama/llama3.2-1b","qwen/qwen3-0.6b",
            do
                # if [ $i -ge 8 ] && [ $version -eq "v1" ]; then
                #     continue
                # fi
                for client_num in {4,}
                do
                    for lag_ratios_index in {0,}
                    do
                        for qo in  {"fifo","lifo","straggler_fo"} #,"lifo","fifo","straggler_fo"
                        do
                            echo "$j-th Running Server with {model $model ,dataset=$dataset, USFL version=$version,client num =$i,LoRA=False,split_point=2, queue_order=$qo}"
                            python experiment/server_run.py -NC=${client_num} -V=${version} -SP=3 -M=${model} -P=${port} -CKPT -DS=${dataset} -LAG="${lag_ratios_index}" -QO=${qo}
                            port=$((port + 1))
                        done
                    done
                done
            done
        done
    done
done