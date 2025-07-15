from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import LlamaPreTrainedModel, LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

# from transformers.modeling_attn_mask_utils import
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel
from usfl.llm.split_config import SplitModelConfig


class LlamaServer(PreTrainedModel):

    def __init__(self, config: LlamaConfig, split_config: SplitModelConfig):
        super().__init__(config)
        self.split_config = split_config

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: LlamaForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[LlamaDecoderLayer]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            hidden_layers[i].self_attn.layer_idx = i
            self.layers.append(hidden_layers[i])

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: LlamaForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[LlamaDecoderLayer]
        self.layers = nn.ModuleList([LlamaDecoderLayer(self.config, layer_idx + to_l) for layer_idx in range(to_l - from_l)])
        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())

    def load_from_pretrained_model(self, pretrained_model: LlamaForCausalLM, logical=True):
        from_l = self.split_config.head_layer_num
        to_l = self.split_config.head_layer_num + self.split_config.server_layer_num
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,  # 来自head的输出隐藏状态
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 来自head model的输出

        for decoder_layer in self.layers:
            # print("hidden_states:", hidden_states)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=None,  # 这个在源码里没有用到
                use_cache=False,
                output_attentions=False,
            )

            hidden_states = layer_outputs[0]

        return hidden_states


def load_llama_server(pretrained_model: LlamaForCausalLM, split_config: SplitModelConfig) -> LlamaServer:
    config = pretrained_model.config
    if split_config.server_layer_num <= 0:
        split_config.server_layer_num = config.num_hidden_layers - split_config.head_layer_num - split_config.tail_layer_num
    llama_server = None
    # load from pretrained gpt2 model
    llama_server = LlamaServer(config, split_config)
    llama_server.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)
    return llama_server
