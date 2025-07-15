import os
import shutil

from typing import List, Tuple, Optional
from peft import PeftModel, PeftModelForCausalLM
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers import PreTrainedModel

from usfl.llm.split_config import SplitModelConfig


class GPT2ClientHead(PreTrainedModel):

    def __init__(self, config: GPT2Config, split_config: SplitModelConfig):
        super().__init__(config)
        self.split_config = split_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self._attn_implementation = config._attn_implementation

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: GPT2LMHeadModel, from_l, to_l):
        self.embed_dim = pretrained_model.config.hidden_size
        self.wte = pretrained_model.transformer.wte
        self.wpe = pretrained_model.transformer.wpe
        self.drop = pretrained_model.transformer.drop
        hidden_layers = pretrained_model.transformer.h
        hidden_layers: List[GPT2Block]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            self.layers.append(hidden_layers[i])

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: GPT2LMHeadModel, from_l, to_l):
        self.embed_dim = pretrained_model.config.hidden_size
        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(self.config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(self.config.embd_pdrop)
        hidden_layers = pretrained_model.transformer.h
        hidden_layers: List[GPT2Block]
        self.layers = nn.ModuleList([GPT2Block(self.config, layer_idx) for layer_idx in range(to_l - from_l)])
        self.wte.load_state_dict(pretrained_model.transformer.wte.state_dict())
        self.wpe.load_state_dict(pretrained_model.transformer.wpe.state_dict())
        self.drop.load_state_dict(pretrained_model.transformer.drop.state_dict())
        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())

    def load_from_pretrained_model(self, pretrained_model: GPT2LMHeadModel, logical=True):
        from_l = 0
        to_l = self.split_config.head_layer_num
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)

    # TODO 将三个类进行分成子类和父类，将公共方法合并
    def save_pretrained(self, save_directory, model_prefix="model"):
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
        config = GPT2Config.from_pretrained(dir)
        from_l = 0
        to_l = split_config.head_layer_num
        gpt2_head = GPT2ClientTail(config, split_config)
        gpt2_head.add_module("wte", nn.Embedding(config.vocab_size, config.n_embd))
        gpt2_head.add_module("wpe", nn.Embedding(config.max_position_embeddings, config.n_embd))
        gpt2_head.add_module("drop", nn.Dropout(config.embd_pdrop))
        gpt2_head.add_module("layers", nn.ModuleList([GPT2Block(config, layer_idx) for layer_idx in range(to_l - from_l)]))
        state_dict_pth = os.path.join(dir, f'{model_prefix}.pth')
        state_dict = torch.load(state_dict_pth, map_location=map_location, weights_only=True)
        gpt2_head.load_state_dict(state_dict, strict=strict)
        return gpt2_head

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
        from_l = 0
        to_l = split_config.head_layer_num
        gpt2_head = GPT2ClientTail(config, split_config)
        gpt2_head.add_module("wte", nn.Embedding(config.vocab_size, config.n_embd))
        gpt2_head.add_module("wpe", nn.Embedding(config.max_position_embeddings, config.n_embd))
        gpt2_head.add_module("drop", nn.Dropout(config.embd_pdrop))
        gpt2_head.add_module("layers", nn.ModuleList([GPT2Block(config, layer_idx) for layer_idx in range(to_l - from_l)]))
        gpt2_head = PeftModel.from_pretrained(gpt2_head, save_dir, is_trainable=True)
        return gpt2_head
        pass

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # 检查输入
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 处理 token_type_ids
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        # 处理 position_ids
        if position_ids is None:
            position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=device).unsqueeze(0)

        # 处理输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # 处理 attention_mask
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=attention_mask,
            input_shape=(batch_size, input_shape[-1]),
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )
        # 处理 head_mask
        head_mask = self.get_head_mask(None, self.config.n_layer)  # 不需要传入 head_mask 参数

        # 添加 token_type_ids 嵌入
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        # Dropout
        hidden_states = self.drop(hidden_states)

        # 前向传播每一层
        for i in range(len(self.layers)):
            block = self.layers[i]
            outputs = block.forward(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=False,
            )
            hidden_states = outputs[0]

        return hidden_states, attention_mask


class GPT2ClientTail(PreTrainedModel):

    def __init__(self, config: GPT2Config, split_config: SplitModelConfig):
        super().__init__(config)
        self.split_config = split_config

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: GPT2LMHeadModel, from_l, to_l):
        self.embed_dim = pretrained_model.config.n_embd
        hidden_layers = pretrained_model.transformer.h
        hidden_layers: List[GPT2Block]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            self.layers.append(hidden_layers[i])
        self.ln_f = pretrained_model.transformer.ln_f
        self.lm_head = pretrained_model.lm_head

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: GPT2LMHeadModel, from_l, to_l):
        self.embed_dim = pretrained_model.config.n_embd
        hidden_layers = pretrained_model.transformer.h
        hidden_layers: List[GPT2Block]
        self.layers = nn.ModuleList([GPT2Block(self.config, layer_idx + from_l) for layer_idx in range(to_l - from_l)])
        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=pretrained_model.config.layer_norm_epsilon)
        self.ln_f.load_state_dict(pretrained_model.transformer.ln_f.state_dict())
        self.lm_head = nn.Linear(self.embed_dim, self.config.vocab_size, bias=False)
        self.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())

    def load_from_pretrained_model(self, pretrained_model: GPT2LMHeadModel, logical=True):
        from_l = self.split_config.total_hidden_layers - self.split_config.tail_layer_num
        to_l = self.split_config.total_hidden_layers
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)

    def save_pretrained(self, save_directory, model_prefix="model"):
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
        config = GPT2Config.from_pretrained(dir)
        from_l = split_config.total_hidden_layers - split_config.tail_layer_num
        to_l = split_config.total_hidden_layers
        gpt2_tail = GPT2ClientTail(config, split_config)
        gpt2_tail.add_module("layers", nn.ModuleList([GPT2Block(config, layer_idx + from_l) for layer_idx in range(to_l - from_l)]))
        gpt2_tail.add_module("ln_f", nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon))
        gpt2_tail.add_module("lm_head", nn.Linear(config.n_embd, config.vocab_size, bias=False))
        state_dict_pth = os.path.join(dir, f'{model_prefix}.pth')
        state_dict = torch.load(state_dict_pth, map_location=map_location, weights_only=True)
        gpt2_tail.load_state_dict(state_dict, strict=strict)
        return gpt2_tail

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
        from_l = split_config.total_hidden_layers - split_config.tail_layer_num
        to_l = split_config.total_hidden_layers
        gpt2_tail = GPT2ClientTail(config, split_config)
        gpt2_tail.add_module("layers", nn.ModuleList([GPT2Block(config, layer_idx + from_l) for layer_idx in range(to_l - from_l)]))
        gpt2_tail.add_module("ln_f", nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon))
        gpt2_tail.add_module("lm_head", nn.Linear(config.n_embd, config.vocab_size, bias=False))
        gpt2_tail = PeftModel.from_pretrained(gpt2_tail, save_dir, is_trainable=True)
        return gpt2_tail
        pass

    def forward(
        self,
        hidden_status_from_server: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        lm_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if head_mask is None:
            head_mask = [None] * self.config.n_layer
        hidden_states = hidden_status_from_server
        for i in range(len(self.layers)):
            block = self.layers[i]
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=False,
            )

            hidden_states = outputs[0]

        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
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

        return CausalLMOutputWithCrossAttentions(loss=loss, logits=lm_logits)


def load_gpt_client_models(pretrained_model: GPT2LMHeadModel, split_config: SplitModelConfig) -> Tuple[GPT2ClientHead, GPT2ClientTail]:
    config = pretrained_model.config
    if split_config.server_layer_num <= 0:
        split_config.server_layer_num = config.n_layer - split_config.tail_layer_num - split_config.head_layer_num
    head_model = GPT2ClientHead(config, split_config)
    head_model.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)
    tail_model = GPT2ClientTail(config, split_config)
    tail_model.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)
    return head_model, tail_model
