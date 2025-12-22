#!/bin/bash

port=9110
# 定义任务配置： "数据集名 序列长度"
# 这样写可以合并两个大循环，避免代码重复
# configurations=("gsm8k 256" "dialogsum 512" "e2e 128")
configurations=("dialogsum 512")

for j in {1,} 
do
    # 遍历配置数组
    for config in "${configurations[@]}"
    do
        # 读取当前配置 (set -- 将字符串拆分为参数 $1, $2)
        set -- $config
        dataset=$1
        max_seq_len=$2

        for version in "v3"
        do
            # 确保这里的模型列表与 Server 端完全一致
            for model in "qwen/qwen3-1.7b" "meta-llama/llama3.2-1b" #"qwen/qwen3-0.6b" "meta-llama/llama3.2-1b" "qwen/qwen3-1.7b"
            do
                # 动态设置 split_point (SP)
                case "$model" in
                    "qwen/qwen3-0.6b")      current_sp=4 ;;
                    "meta-llama/llama3.2-1b") current_sp=3 ;;
                    "qwen/qwen3-1.7b")      current_sp=4 ;;
                    *)                      current_sp=3 ;; # 默认值
                esac
                
                for client_num in 32 1 2 4 8 16 
                do
                    for lag_ratios_index in 0 # 0:无异质性
                    do
                        for qo in "fifo"
                        do
                            echo "$j-th Client: Model=$model, Data=$dataset, Len=$max_seq_len, SP=$current_sp, Ver=$version, Clients=$client_num, Port=$port"
                            
                            python experiment/client_run.py \
                                -NC=${client_num} \
                                -V=${version} \
                                -L \
                                -SP=${current_sp} \
                                -M=${model} \
                                -P=${port} \
                                -B=4 \
                                -DS=${dataset} \
                                -SL=${max_seq_len} \
                                -LAG="${lag_ratios_index}" \
                                -QO=${qo} 
                            
                            # 端口自增
                            port=$((port + 1))
                            
                            echo "----------------------------------------"
                            # cd vis/
                            # python dcp.py ...
                            # cd ..
                        done
                    done
                done
            done
        done
    done
done

# batch_sizes=(4 6 8 6)
# max_seq_len=128
# port=9111
# for j in {1,} 
# do
#     for dataset in "gsm8k"
#     do
#         for version in "merge"
#         do
#             for model in "meta-llama/llama3.2-1b" #"meta-llama/llama3.2-1b"
#             do
#                 for client_num in 24
#                 do
#                     for lag_ratios_index in 2 3  # 0:无异质性， 1:轻度异质性， 2:中度异质性， 3:高度异质性
#                     do
#                         current_batch=${batch_sizes[$lag_ratios_index]}

#                         for qo in "fifo" # "fifo","lifo","straggler_fo"
#                         do
#                             echo "$j-th Running Clients with {model $model ,dataset $dataset, USFL version=$version, client num $i, max_seq_len=$max_seq_len, LoRA=True, split_point=3}"
#                             python experiment/client_run.py -NC=${client_num} -V=${version} -L -SP=3 -M=${model} -P=${port} -B=${current_batch} -DS=${dataset} -SL=${max_seq_len} -LAG="${lag_ratios_index}" -QO=${qo} -SR 0.2
#                             port=$((port + 1))
#                             echo "----------------------------------------"
#                             cd vis/
#                             # python dcp.py -V=${version} -LAG="${lag_ratios_index}" -NC=${client_num} -QO=${qo}
#                             # python vis.py -V=${version} -LAG="${lag_ratios_index}" -NC=${client_num} -QO=${qo}
#                             echo "----------------------------------------"
#                             cd ..
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done
