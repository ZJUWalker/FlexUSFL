import torch
import torch.nn as nn
from typing import Dict, Any
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from usfl.llm import (
    load_gpt_server_model,
    load_llama_server,
    load_qwen3_server,
    SplitModelConfig,
)


def get_model_layer_num(model_dir: str) -> int:
    config = AutoConfig.from_pretrained(model_dir)
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "n_layer"):
        return config.n_layer
    else:
        raise ValueError("Cannot find layer number")


def load_server_model(
    model_dir: str,
    server_args: Dict[str, Any],
    split_point: int,
) -> nn.Module:
    if server_args["use_qlora_4bit"] or server_args["use_qlora_8bit"]:
        ql_config = BitsAndBytesConfig(
            load_in_4bit=server_args["use_qlora_4bit"],
            load_in_8bit=server_args["use_qlora_8bit"],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        ql_config = None

    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=ql_config, device_map="cpu")
    model.train()
    if server_args["use_qlora_4bit"] or server_args["use_qlora_8bit"]:
        model = prepare_model_for_kbit_training(model)
        if server_args["use_lora"]:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    split_config = SplitModelConfig(
        head_layer_num=split_point,
        tail_layer_num=split_point,
    )

    if "gpt" in server_args["model"].lower():
        server = load_gpt_server_model(model, split_config)
    elif "llama" in server_args["model"].lower():
        server = load_llama_server(model, split_config)
    elif "qwen" in server_args["model"].lower():
        server = load_qwen3_server(model, split_config)
    else:
        raise ValueError(f"unsupported model card {server_args['model']}")

    return server
