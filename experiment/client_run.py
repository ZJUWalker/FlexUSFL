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


def client_worker(rank: int, args: dict):
    set_seed(SEED)
    dataset_name = args["dataset"]
    num_clients = args["num_clients"]
    batch_size = args["batch_size"]
    max_seq_len = args["max_seq_len"]
    model_name = args["model"]
    model_dir = os.path.join("/share/models", model_name)
    split_point = args["split_point"]
    device = f"cuda:{rank % 3}"  # use 3 gpus
    log_dir = f"log/loss/{args['model']}/client_number_{args['num_clients']}/{args['version']}/client"
    logger = create_logger(log_file_name=f"client_{rank}.log", console_output=False, log_dir=log_dir)
    logger.info(f"client {rank} start with args: {args}")
    # ---------------load model and tokenizer --------------------------
    head, tail, tokenizer = load_client(model_dir, args, split_point)
    # -----------------load dataset------------------------------------
    client_dataloaders = load_dataset(dataset_name, tokenizer, list(range(num_clients)), batch_size, max_seq_len)
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
    )
    client.train_epoch()


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
    parser.add_argument("-BPS", "--batch_per_sync", type=int, default=20)
    parser.add_argument("-DS", "--dataset", type=str, default="gsm8k")
    parser.add_argument("-E", "--epoch", type=int, default=1)
    parser.add_argument("-SP", "--split_point", type=int, default=2)
    parser.add_argument("-LR", "--learning_rate", type=float, default=5e-4)
    client_args = parser.parse_args()
    client_args = vars(client_args)
    num_clients = client_args["num_clients"]
    # mp.set_start_method("spawn", force=True)
    print("create client processes")

    mp.spawn(
        client_worker,
        args=(client_args,),
        nprocs=num_clients,
        join=True,
    )
    print("client processes done")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
