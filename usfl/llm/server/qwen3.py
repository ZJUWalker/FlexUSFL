from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM, Qwen3Model, Qwen3PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RotaryEmbedding,
    Qwen3RMSNorm,
)
from usfl.llm.split_config import SplitModelConfig


class Qwen3Server(PreTrainedModel):

    def __init__(self, config: Qwen3Config, split_config: SplitModelConfig):
        super().__init__(config)
        self.split_config = split_config
        self.layers = None

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: Qwen3ForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Qwen3DecoderLayer]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            hidden_layers[i].self_attn.layer_idx = i
            self.layers.append(hidden_layers[i])

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: Qwen3ForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Qwen3DecoderLayer]
        self.layers = nn.ModuleList([Qwen3DecoderLayer(self.config, layer_idx + to_l) for layer_idx in range(to_l - from_l)])
        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())

    def load_from_pretrained_model(self, pretrained_model: Qwen3ForCausalLM, logical=True):
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


def load_qwen3_server(pretrained_model: Qwen3ForCausalLM, split_config: SplitModelConfig) -> Qwen3Server:
    config = pretrained_model.config
    if split_config.server_layer_num <= 0:
        split_config.server_layer_num = config.num_hidden_layers - split_config.head_layer_num - split_config.tail_layer_num
    qwen3_server = None
    # load from pretrained gpt2 model
    qwen3_server = Qwen3Server(config, split_config)
    qwen3_server.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)
    return qwen3_server
