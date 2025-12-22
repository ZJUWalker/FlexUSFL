#!/bin/bash

port=9110
# 与 Client 端保持一致的配置顺序
# configurations=("gsm8k" "dialogsum" "e2e") 
configurations=("dialogsum")
# Server端脚本似乎不需要 max_seq_len (-SL)，如果需要请参照 Client 端修改

for j in {1,}
do
    for dataset in "${configurations[@]}"
    do
        for version in "v3"
        do
            for model in "qwen/qwen3-1.7b" "meta-llama/llama3.2-1b" # "qwen/qwen3-0.6b" "meta-llama/llama3.2-1b" "qwen/qwen3-1.7b"
            do
                # 动态设置 split_point
                case "$model" in
                    "qwen/qwen3-0.6b")      current_sp=4 ;;
                    "meta-llama/llama3.2-1b") current_sp=3 ;;
                    "qwen/qwen3-1.7b")      current_sp=4 ;;
                    *)                      current_sp=3 ;;
                esac
                
                for client_num in 32 1 2 4 8 16 
                do
                    for lag_ratios_index in 0
                    do
                        for qo in "fifo"
                        do
                            echo "$j-th Server: Model=$model, Data=$dataset, Ver=$version, Clients=$client_num, SP=$current_sp, Port=$port"
                            
                            python experiment/server_run.py \
                                -NC=${client_num} \
                                -V=${version} \
                                -SP=${current_sp} \
                                -M=${model} \
                                -P=${port} \
                                -CKPT \
                                -DS=${dataset} \
                                -LAG="${lag_ratios_index}" \
                                -QO=${qo}
                            
                            # 端口自增
                            port=$((port + 1))
                        done
                    done
                done
            done
        done
    done
done


# port=9111
# for j in {1,}
# do
#     for dataset in "gsm8k"
#     do
#         for version in "merge"
#         do
#             for model in "meta-llama/llama3.2-1b" #"meta-llama/llama3.2-1b","qwen/qwen3-0.6b",
#             do
#                 for client_num in 24
#                 do
#                     for lag_ratios_index in 2 3 # 0:无异质性， 1:轻度异质性， 2:中度异质性， 3:高度异质性
#                     do
#                         for qo in "fifo" #,"lifo","fifo","straggler_fo"
#                         do
#                             echo "$j-th Running Server with {model $model ,dataset=$dataset, USFL version=$version,client num =$i,LoRA=False,split_point=3, queue_order=$qo}"
#                             python experiment/server_run.py -NC=${client_num} -V=${version} -SP=3 -M=${model} -P=${port} -CKPT -DS=${dataset} -LAG="${lag_ratios_index}" -QO=${qo}
#                             port=$((port + 1))
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done