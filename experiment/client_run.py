import os
import argparse
import warnings

import torch
import torch.multiprocessing as mp
import torch
from usfl.client import Client
from usfl.utils.log_utils import create_logger
from usfl.utils.load_utils import load_client, load_dataset
from usfl.utils.exp import set_seed

SEED = 0
warnings.filterwarnings("ignore", message="To copy construct from a tensor", category=UserWarning)


def client_worker(rank: int, args: dict, lag_ratios: list, barrier):
    set_seed(SEED)
    dataset_name = args["dataset"]
    num_clients = args["num_clients"]
    base_batch_size = args["batch_size"]
    max_seq_len = args["max_seq_len"]
    model_name = args["model"]
    model_dir = os.path.join("/share/models", model_name)
    split_point = args["split_point"]
    available_gpus = [1, 2, 3, 4, 5, 6, 7]
    device = f"cuda:{available_gpus[rank % len(available_gpus)]}"

    lag_ratio = lag_ratios[rank % len(lag_ratios)]
    if args["version"] == "merge":
        min_lag = min(lag_ratios)
        calculated_batch_size = base_batch_size * (min_lag / lag_ratio)
        batch_size = max(1, int(calculated_batch_size))

        print(f"base batch size: {base_batch_size}, lag_ratio: {lag_ratio}, calculated_batch_size: {calculated_batch_size}")
    else:
        batch_size = base_batch_size
    # batch_size = base_batch_size
    if args["mode"] == "hetero":
        root_name = "hetero"
    elif args["mode"] == "main":
        root_name = "exp_main"
    else:
        raise ValueError(f"Unknown mode: {args['mode']}")

    log_dir = os.path.join(
        "log",
        root_name,
        args["version"],
        args["model"],
        args["dataset"],
        f"lag_{args['lag_ratio']}",
        f"client_number_{args['num_clients']}",
        args["queue_order"],
        "client",
    )
    logger = create_logger(log_file_name=f"client_{rank}.log", console_output=False, log_dir=log_dir)
    logger.info(f"client {rank} start with args: {args}")
    # ---------------load model and tokenizer --------------------------
    head, tail, tokenizer = load_client(model_dir, args, split_point)
    # -----------------load dataset------------------------------------

    # print(f"client {rank} load dataset")
    if args["partition_mode"] == "exclusive":
        client_dataloaders = load_dataset(dataset_name, tokenizer, list(range(num_clients)), batch_size, max_seq_len, "exclusive")
    else:
        client_dataloaders = load_dataset(
            dataset_name,
            tokenizer,
            list(range(num_clients)),
            batch_size,
            max_seq_len,
            partition_mode="random_overlap",
            sample_ratio=args["sample_ratio"],
        )

    data = client_dataloaders[rank]

    min_batch_num = min([len(cd["train"]) for cd in client_dataloaders.values()])
    if rank == 0:
        print(f"min_batch_num: {min_batch_num}")
    # -----------------create client----------
    # print(f'cuda memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB')
    client = Client(
        client_args=args,
        client_id=rank,
        head_model=head,
        tail_model=tail,
        tokenizer=tokenizer,
        client_device=device,
        train_logger=logger,
        dataset_train=data["train"],
        dataset_test=data["test"] if "test" in data else None,
        batch_num=min_batch_num,
        lag_ratio=lag_ratio,
    )
    # barrier.wait()

    client.train_epoch(barrier=barrier)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=int, default=8000, help="port to listen")
    parser.add_argument("-CK", "--use_sl_ckpt", action="store_true", help="use split l with checkpoint")
    parser.add_argument("-L", "--use_lora", action="store_true", help="use lora")
    parser.add_argument("-Q4", "--use_qlora_4bit", action="store_true", help="use qlora 4bit")
    parser.add_argument("-Q8", "--use_qlora_8bit", action="store_true", help="use qlora 8bit")
    parser.add_argument("-M", "--model", type=str, default="meta-llama/llama3.2-1b", help="model card")
    parser.add_argument("-B", "--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("-SL", "--max_seq_len", type=int, default=256, help="max sequence length")
    parser.add_argument("-NC", "--num_clients", type=int, default=2)
    parser.add_argument("-S", "--step", type=int, default=20)
    parser.add_argument("-V", "--version", type=str, default="v1")
    parser.add_argument("-BPS", "--batch_per_sync", type=int, default=20, help="batch per sync")
    parser.add_argument("-DS", "--dataset", type=str, default="gsm8k")
    parser.add_argument("-E", "--epoch", type=int, default=1)
    parser.add_argument("-SP", "--split_point", type=int, default=3)
    parser.add_argument("-LR", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("-LAG", "--lag_ratio", type=int, default=0, help="simulate client computation lag by multiplying this ratio")
    parser.add_argument("-QO", "--queue_order", type=str, default="fifo", help="queue order for clients")
    parser.add_argument(
        "-PM", "--partition_mode", type=str, default="random_overlap", help="partition mode for clients(exclusive or random_overlap)"
    )
    parser.add_argument("-SR", "--sample_ratio", type=float, default=0.2, help="sample ratio for random partition mode")
    parser.add_argument("--mode", type=str, default="main", help="main or hetero")

    client_args = parser.parse_args()
    client_args = vars(client_args)
    num_clients = client_args["num_clients"]
    print(f"batch size per client: {client_args['batch_size']}, total batch size: {client_args['batch_size']*num_clients}")
    # mp.set_start_method("spawn", force=True)
    print("create client processes")

    # simulate lag
    lag_ratios_list = [
        [1.0, 1.0, 1.0, 1.0],  # 无异质性
        [1.0, 1.2, 1.4, 1.6],  # 轻度异质性
        [1.0, 2.0, 3.0, 4.0],  # 中度异质性
        [1.0, 1.0, 2.0, 10.0],  # 严重异质性
        [2, 4, 6, 8],
        [3, 6, 9, 12],
        # [1.0, 1.25, 1.5, 1.75],  # 3
        # [1.0, 1.5, 2.0, 2.5],
        # [1.0, 2.0, 3.0, 4.0],
        # [1.0, 1.0, 1.0, 7.0],
        # [2, 4, 6, 8],  # 7
        # [4, 8, 12, 16],
        # [1.0, 4.0, 8.0],
    ]
    lag_ratios = lag_ratios_list[client_args["lag_ratio"]]
    print(f"lag_ratios for clients: {lag_ratios}")

    barrier = mp.Barrier(num_clients)  # 所有 client 进程在这里同步

    mp.spawn(
        client_worker,
        args=(
            client_args,
            lag_ratios,
            barrier,
        ),
        nprocs=num_clients,
        join=True,
    )
    print("client processes done")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
