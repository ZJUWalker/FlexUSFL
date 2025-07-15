from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import LlamaPreTrainedModel, LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

# from transformers.modeling_attn_mask_utils import
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel

from usfl.llm.split_config import SplitModelConfig


def get_param_count(model: LlamaPreTrainedModel):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LlamaClientHead(PreTrainedModel):

    def __init__(self, config: LlamaConfig, split_config: SplitModelConfig):
        super().__init__(config)
        self.split_config = split_config
        self.embed_tokens = None
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.all_embeds = None

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: LlamaForCausalLM, from_l, to_l):
        emb_layer = pretrained_model.model.embed_tokens
        self.embed_tokens = emb_layer
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[LlamaDecoderLayer]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            hidden_layers[i].self_attn.layer_idx = i
            self.layers.append(hidden_layers[i])
        self.rotary_emb = pretrained_model.model.rotary_emb

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: LlamaForCausalLM, from_l, to_l):
        self.embed_tokens = nn.Embedding(pretrained_model.config.vocab_size, pretrained_model.config.hidden_size, self.padding_idx)
        emb_layer = pretrained_model.model.embed_tokens
        self.embed_tokens.load_state_dict(emb_layer.state_dict())
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[LlamaDecoderLayer]
        self.layers = nn.ModuleList([LlamaDecoderLayer(self.config, layer_idx) for layer_idx in range(to_l - from_l)])
        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())
        self.rotary_emb = LlamaRotaryEmbedding(config=pretrained_model.config)
        self.rotary_emb.load_state_dict(pretrained_model.model.rotary_emb.state_dict())

    def load_from_pretrained_model(self, pretrained_model: LlamaForCausalLM, logical=True):
        from_l = 0
        to_l = self.split_config.head_layer_num
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple:
        # --------------------------------------------------------------------------处理输入tokens
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        causal_mask = _update_causal_mask(self, attention_mask, inputs_embeds, position_ids, False)
        # 输入embedding
        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # 前向传播
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
            )

            hidden_states = layer_outputs[0]
        return (
            hidden_states,
            causal_mask,
            position_embeddings,
        )  # 因为对attention_mask的处理，所以这里返回causal_mask，为后续server和tail的前向传播做准备


class LlamaClientTail(PreTrainedModel):
    def __init__(self, config: LlamaConfig, split_config: SplitModelConfig):
        super().__init__(config)
        self.split_config = split_config

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: LlamaForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[LlamaDecoderLayer]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            if not self.split_config.with_server:
                hidden_layers[i].self_attn.layer_idx = i - self.split_config.server_layer_num
            else:
                hidden_layers[i].self_attn.layer_idx = i
            self.layers.append(hidden_layers[i])
        # 加载最后的Norm和lm_head
        self.norm = pretrained_model.model.norm
        self.lm_head = pretrained_model.lm_head

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: LlamaForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[LlamaDecoderLayer]
        self.layers = nn.ModuleList([LlamaDecoderLayer(self.config, layer_idx + to_l) for layer_idx in range(to_l - from_l)])
        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())
        self.norm = LlamaRMSNorm(pretrained_model.config.hidden_size, eps=pretrained_model.config.rms_norm_eps)
        self.norm.load_state_dict(pretrained_model.model.norm.state_dict())
        self.lm_head = nn.Linear(pretrained_model.config.hidden_size, pretrained_model.config.vocab_size, bias=False)
        self.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())

    def load_from_pretrained_model(self, pretrained_model: LlamaForCausalLM, logical=True):
        from_l = self.split_config.head_layer_num + self.split_config.server_layer_num
        to_l = self.split_config.total_hidden_layers
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,  # 来自head的输出隐藏状态
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        labels: Optional[torch.LongTensor] = None,
        lm_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=None,
                use_cache=False,
                output_attentions=False,
            )

            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            if lm_mask is not None:
                shift_lm_mask = lm_mask[..., 1:].contiguous()
                shift_lm_mask = shift_lm_mask.view(-1)
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * shift_lm_mask.float()
                loss = loss.sum() / shift_lm_mask.sum()
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


def _update_causal_mask(
    partitioned_model: LlamaPreTrainedModel,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position_ids: torch.Tensor = None,
    output_attentions: bool = False,
):
    # TODO 简化_update_causal_mask的过程，抛去训练过程中不需要的部分
    # attention impl is spda
    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = 0
    # using_static_cache = isinstance(past_key_values, StaticCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if partitioned_model.config._attn_implementation == "sdpa":
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=partitioned_model.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1

    if attention_mask is not None and attention_mask.dim() == 4:
        # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
        if attention_mask.max() != 0:
            raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position_ids.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
    if (
        partitioned_model.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    # print(causal_mask.shape)
    return causal_mask


# 将llama模型分割成三个部分，head，server，tail
# 首先从与训练模型中加载模型
def load_llama_client(pretrained_model: LlamaForCausalLM, split_config: SplitModelConfig) -> Tuple[LlamaClientHead, LlamaClientTail]:
    config = pretrained_model.config
    if split_config.server_layer_num <= 0:
        split_config.server_layer_num = config.num_hidden_layers - split_config.head_layer_num - split_config.tail_layer_num
    head_model = LlamaClientHead(config, split_config)
    head_model.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)
    tail_model = LlamaClientTail(config, split_config)
    tail_model.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)
    return head_model, tail_model
