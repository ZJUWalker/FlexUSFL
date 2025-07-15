import os
from typing import List, Optional
import torch
import torch.nn as nn
from peft import PeftModel, PeftModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Config, AutoConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from transformers.utils import (
    logging,
)
from transformers import PreTrainedModel

from usfl.llm.split_config import SplitModelConfig

logger = logging.get_logger(__name__)


class GPT2Server(PreTrainedModel):

    def __init__(self, config: GPT2Config, split_config: SplitModelConfig):
        super().__init__(config)
        self.split_config = split_config

    def get_hidden_state_dim(self):
        return self.config.n_embd

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: GPT2LMHeadModel, from_l, to_l):
        hidden_layers = pretrained_model.transformer.h
        hidden_layers: List[GPT2Block]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            self.layers.append(hidden_layers[i])

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: GPT2LMHeadModel, from_l, to_l):
        hidden_layers = pretrained_model.transformer.h
        hidden_layers: List[GPT2Block]
        self.layers = nn.ModuleList([GPT2Block(self.config, layer_idx + from_l) for layer_idx in range(to_l - from_l)])
        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())

    def load_from_pretrained_model(self, pretrained_model: GPT2LMHeadModel, logical=True):
        from_l = self.split_config.head_layer_num
        to_l = self.split_config.head_layer_num + self.split_config.server_layer_num
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)

    # TODO 将三个类进行分成子类和父类，将公共方法合并
    def save_pretrained(self, save_directory: str, model_prefix="model"):
        # assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        state_dict = self.state_dict()
        # save tail model
        torch.save(state_dict, os.path.join(save_directory, f'{model_prefix}.pth'))
        # Save the config
        self.config.save_pretrained(save_directory)
        self.split_config.save_pretrained(save_directory)

    @staticmethod
    def from_pretrained(dir: str, model_prefix='model', map_location='cpu', strict=True):
        split_config = SplitModelConfig.load_pretrained(dir)
        config = AutoConfig.from_pretrained(dir)
        from_l = split_config.head_layer_num
        to_l = split_config.head_layer_num + split_config.server_layer_num
        gpt2_server = GPT2Server(config, split_config)
        gpt2_server.add_module("layers", nn.ModuleList([GPT2Block(config, layer_idx + from_l) for layer_idx in range(to_l - from_l)]))
        state_dict_pth = os.path.join(dir, f'{model_prefix}.pth')
        state_dict = torch.load(state_dict_pth, map_location=map_location, weights_only=True)
        gpt2_server.load_state_dict(state_dict, strict=strict)
        return gpt2_server

    @staticmethod
    def save_lora_model(peft_model: PeftModelForCausalLM, save_dir: str):
        original_model = peft_model.base_model.model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        peft_model.save_pretrained(save_dir)
        original_model.config.save_pretrained(save_dir)
        if hasattr(original_model, 'split_config'):
            original_model.split_config.save_pretrained(save_dir)

    @staticmethod
    def load_lora_model(save_dir: str):
        split_config = SplitModelConfig.load_pretrained(save_dir)
        config = GPT2Config.from_pretrained(save_dir)
        from_l = split_config.head_layer_num
        to_l = split_config.head_layer_num + split_config.server_layer_num
        gpt2_server = GPT2Server(config, split_config)
        gpt2_server.add_module("layers", nn.ModuleList([GPT2Block(config, layer_idx + from_l) for layer_idx in range(to_l - from_l)]))
        gpt2_server = PeftModel.from_pretrained(gpt2_server, save_dir, is_trainable=True)
        return gpt2_server
        pass

    def forward(
        self,
        hidden_status_from_head: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if head_mask is None:
            head_mask = [None] * self.config.n_layer
        hidden_states = hidden_status_from_head
        for i in range(len(self.layers)):
            block = self.layers[i]
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=False,
            )
            hidden_states = outputs[0]
        return hidden_states  # only need to return the last hidden_states in the training phase


# load_gpt_server_model
def load_gpt_server_model(pretrained_model: GPT2LMHeadModel, split_config: SplitModelConfig) -> GPT2Server:
    config = pretrained_model.config
    if split_config.server_layer_num <= 0:
        split_config.server_layer_num = config.n_layer - split_config.tail_layer_num - split_config.head_layer_num
    if split_config.server_layer_num <= 0:
        raise ValueError("server_layer_num should be greater than 0")
    server_model = None
    # load from pretrained gpt2 model
    server_model = GPT2Server(config, split_config)
    server_model.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)
    return server_model
