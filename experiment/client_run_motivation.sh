
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
            for i in {1,2,3,4}
            do
                echo "Running Clients with {model $model ,dataset $dataset, USFL version=$version, client num $i, LoRA=True,split_point=2}"
                python experiment/client_run.py -NC=${i} -V=${version} -L -SP=2 -M=${model} -P=${port} -B=1 -DS=${dataset}
                port=$((port + 1))
            done
        done
    done
done