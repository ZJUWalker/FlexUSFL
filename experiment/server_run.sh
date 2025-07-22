#!/bin/bash

port=8000
# for version in {"v3",}
# do
#     for i in {4,6,8}
#     do
#         for model in {"qwen/qwen3-1.7b",}
#         do
#             echo "Running Server with {model $model ,USFL version=$version,client num =$i,LoRA=False,split_point=2}"
#             python experiment/server_run.py -NC=${i} -V=${version} -SP=2 -M=${model} -P=${port}
#             port=$((port + 1))
#         done
#     done
# done
model=qwen/qwen3-1.7b
version=v3
i=48
for bs in {1}
do
    echo "Running Server with {model $model ,USFL version=$version,client num =$i,LoRA=False,split_point=2}"
    python experiment/server_run.py -NC=${i} -V=${version} -SP=2 -M=${model} -P=${port}
    port=$((port + 1))
done