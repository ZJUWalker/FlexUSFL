#!/bin/bash

port=8009
max_seq_len=128

# lag_ratios_index=0

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
        for version in {"v3",}
        do
            for model in {"meta-llama/llama3.2-1b",} #"meta-llama/llama3.2-1b"
            do
                # if [ $i -ge 8 ] && [ $version -eq "v1" ]; then
                #     continue
                # fi
                for i in {8,}
                do
                    for lag_ratios_index in {5,}
                    do
                        for qo in  {"fifo","lifo"}
                        do
                            echo "$j-th Running Clients with {model $model ,dataset $dataset, USFL version=$version, client num $i, max_seq_len=$max_seq_len, LoRA=True, split_point=2}"
                            python experiment/client_run.py -NC=${i} -V=${version} -L -SP=2 -M=${model} -P=${port} -B=1 -DS=${dataset} -SL=${max_seq_len} -LAG="${lag_ratios_index}" -QO=${qo}
                            port=$((port + 1))
                            echo "----------------------------------------"
                            cd vis/
                            python dcp.py -V=${version} -LAG="${lag_ratios_index}" -NC=${i} -QO=${qo}
                            python vis.py -V=${version} -LAG="${lag_ratios_index}" -NC=${i} -QO=${qo}
                            echo "----------------------------------------"
                            cd ..
                        done
                    done
                done
            done
        done
    done
done