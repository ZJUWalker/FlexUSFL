import os
import time
import torch
import argparse
from transformers import AutoConfig
from usfl.utils.exp import set_seed
from usfl.utils.load_utils import *
from usfl.utils.log_utils import create_logger
from usfl.server import *

SEED = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=int, default=8000, help="Port to listen")
    parser.add_argument("-S", "--step", type=int, default=20, help="Number of steps to profile")
    parser.add_argument("-CK", "--use_sl_ckpt", action="store_true", help="Use split learning with checkpoint")
    parser.add_argument("-L", "--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("-Q4", "--use_qlora_4bit", action="store_true", help="Use QLoRA 4-bit")
    parser.add_argument("-Q8", "--use_qlora_8bit", action="store_true", help="Use QLoRA 8-bit")
    parser.add_argument("-M", "--model", type=str, default="meta-llama/llama3.2-1b", help="Model card")
    parser.add_argument("-NC", "--num_clients", type=int, default=1, help="Number of clients")
    parser.add_argument("-BF", "--buffer_size", type=int, default=4096, help="Buffer size for socket communication")
    parser.add_argument("-SD", "--server_device", type=str, default="cuda:3", help="Device for server model")
    parser.add_argument("-CKPT", "--use_checkpoint", action="store_true", help="Use checkpoint")
    parser.add_argument("-SP", "--split_point", type=int, default=2)
    parser.add_argument("-LR", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("-V", "--version", type=str, default="v1", help="usfl version")  # add a flag to specify the version of usfl
    args = parser.parse_args()
    server_args = vars(args)
    set_seed(SEED)
    # =====================================================================
    log_dir = f"log/{server_args['model']}/" f"client_number_{server_args['num_clients']}/{server_args['version']}/server"
    logger = create_logger(log_file_name="training_steps.log", log_dir=log_dir, console_output=False)
    matrix_logger = create_logger(
        log_file_name="training_metrics.log",
        log_dir=log_dir,
        console_output=False,
    )
    matrix_logger.info(f"step | mem alloc(GB) | mem reserved(GB)")
    # =====================================================================
    model_dir = os.path.join("/share/models", server_args["model"])
    split_point = server_args["split_point"]
    server_model = load_server_model(model_dir, server_args, split_point)
    lr = server_args['learning_rate']
    version = server_args["version"]
    if version == "v1":
        server = ServerV1(
            server_args=server_args,
            server_model=server_model,
            server_device=server_args["server_device"],
            num_clients=server_args["num_clients"],
            lr=lr,
            logger=logger,
            matrix_logger=matrix_logger,
        )
    elif version == "v2":
        server = ServerV2(
            server_args=server_args,
            server_model=server_model,
            server_device=server_args["server_device"],
            num_clients=server_args["num_clients"],
            lr=lr,
            logger=logger,
            matrix_logger=matrix_logger,
        )
    else:
        server = ServerV3(
            server_args=server_args,
            server_model=server_model,
            server_device=server_args["server_device"],
            num_clients=server_args["num_clients"],
            lr=lr,
            logger=logger,
            matrix_logger=matrix_logger,
        )
    # =====================================================================
    logger.info(
        "{} server start split point: {}, buffer size: {}B ,use lora {} ,use qlora {} ,use checkpoint {}".format(
            server_args["model"],
            split_point,
            server_args["buffer_size"],
            server_args["use_lora"],
            ("qlora_4bit" if server_args["use_qlora_4bit"] else "qlora_8bit" if server_args["use_qlora_8bit"] else "no_quantization"),
            server_args["use_sl_ckpt"],
        )
    )
    # =====================================================================
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(server_args["server_device"])
    server.run()
