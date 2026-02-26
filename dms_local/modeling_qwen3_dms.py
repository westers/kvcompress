# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg

from .configuration_qwen3_dms import Qwen3Config
from .dms_attention import dms_attention, restore_order
from .dms_cache import DMSCache


torch._dynamo.config.cache_size_limit = 72


def _aux_dms_process_post_attention_data(
    attn_output: torch.Tensor,
    head_dim: int,
    batch: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    seq_len: int,
    q_per_kv: int,
    o_proj: nn.Linear,
    input_shape: tuple[int, int],
):
    attn_output = attn_output.reshape(batch, num_key_value_heads, seq_len, q_per_kv, head_dim).transpose(1, 2)
    attn_output = attn_output.reshape(batch, seq_len, num_attention_heads * head_dim)

    assert input_shape == (batch, seq_len), f"input_shape: {input_shape}"
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = o_proj(attn_output)
    return attn_output


_aux_dms_process_post_attention_data_compiled = torch.compile(_aux_dms_process_post_attention_data)


def _aux_dms_prepare_data_for_attention(
    hidden_states: torch.Tensor,
    decision_scale: float,
    initial_decision_offset: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_proj_fn: nn.Linear,
    k_proj_fn: nn.Linear,
    v_proj_fn: nn.Linear,
    q_norm_fn: nn.Module,
    k_norm_fn: nn.Module,
    head_shape: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, hidden_size = hidden_states.shape
    assert head_shape == (
        batch,
        seq_len,
        -1,
        head_shape[-1],
    ), f"head_shape: {head_shape}"

    query_states = q_norm_fn(q_proj_fn(hidden_states).view(head_shape).transpose(1, 2))
    key_states = k_norm_fn(k_proj_fn(hidden_states).view(head_shape).transpose(1, 2))
    value_states = v_proj_fn(hidden_states).view(head_shape).transpose(1, 2)

    batch, num_attention_heads, seq_len, head_dim = query_states.shape
    num_kv_heads = key_states.shape[1]
    q_per_kv = num_attention_heads // num_kv_heads

    # Use last head dimension of last query head in the gqa group as alpha logit (decision)
    decision_logits = query_states[:, ::q_per_kv, :, -1].clone() * decision_scale - initial_decision_offset

    # True means evict, False means keep
    decisions = (decision_logits > 0).to(decision_logits.dtype)
    assert decisions.shape == (batch, num_kv_heads, seq_len)

    query_states[:, ::q_per_kv, :, -1] = 0
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states[:, ::q_per_kv, :, -1] = 0

    flash_attn_query_states = query_states.reshape(batch * num_kv_heads, q_per_kv, seq_len, head_dim).transpose(1, 2)

    return flash_attn_query_states, query_states, key_states, value_states, decisions


_aux_dms_prepare_data_for_attention_compiled = torch.compile(_aux_dms_prepare_data_for_attention)


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        # DMS specific parameters
        self.dms_alpha_scale = config.dms_alpha_scale
        self.dms_initial_alpha_offset = config.dms_initial_alpha_offset
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.q_per_kv = self.num_attention_heads // self.num_key_value_heads
        self.profiler = None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: DMSCache,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        head_shape = (*input_shape, -1, self.head_dim)
        seq_len = input_shape[-1]
        batch = input_shape[0]

        assert past_key_values is not None, "past_key_values is required for DMS"

        dms_prepare_data_for_attention_fn = (
            _aux_dms_prepare_data_for_attention_compiled if seq_len == 1 else _aux_dms_prepare_data_for_attention
        )

        flash_attn_query_states, query_states, key_states, value_states, decisions = dms_prepare_data_for_attention_fn(
            hidden_states=hidden_states,
            decision_scale=self.dms_alpha_scale,
            initial_decision_offset=self.dms_initial_alpha_offset,
            cos=position_embeddings[0],
            sin=position_embeddings[1],
            q_proj_fn=self.q_proj,
            k_proj_fn=self.k_proj,
            v_proj_fn=self.v_proj,
            q_norm_fn=self.q_norm,
            k_norm_fn=self.k_norm,
            head_shape=head_shape,
        )

        attn_output, restore_order_info = dms_attention(
            new_q=query_states,
            new_q_flash=flash_attn_query_states,
            new_k=key_states,
            new_v=value_states,
            decisions=decisions,
            attn_mask=attention_mask,
            layer_idx=self.layer_idx,
            dms_cache=past_key_values,
            attn_scaling=self.scaling,
            window_size=self.config.dms_window_size,
            profiler=self.profiler,
        )

        dms_process_post_attention_data_fn = (
            _aux_dms_process_post_attention_data_compiled if seq_len == 1 else _aux_dms_process_post_attention_data
        )

        attn_output = dms_process_post_attention_data_fn(
            attn_output=attn_output,
            head_dim=self.head_dim,
            batch=batch,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            seq_len=seq_len,
            q_per_kv=self.q_per_kv,
            o_proj=self.o_proj,
            input_shape=input_shape,
        )

        if restore_order_info is not None:
            # In case of prefill, the data can be reordered
            attn_output = restore_order(attn_output, restore_order_info)

        # DMS does not return attention weights
        attn_weights = None
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Cache offloading attribute (set externally like the profiler pattern)
        self.offload_cache = False

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (use_cache and past_key_values is None) or not isinstance(past_key_values, DMSCache):
            past_key_values = DMSCache(
                dms_window_size=self.config.dms_window_size + 1,
                max_context_length=self.config.max_position_embeddings,
                offloading=self.offload_cache,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        _offloading = getattr(past_key_values, 'offloading', False)

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # Move current layer's cache to GPU before processing (skip during prefill — layers don't exist yet)
            if _offloading and idx < len(past_key_values.layers):
                past_key_values.layers[idx].prefetch()
                torch.cuda.synchronize()  # ensure transfer complete before compute

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # After processing, offload current layer to CPU and pre-move next layer to GPU
            if _offloading and len(past_key_values.layers) > 0:
                past_key_values.layers[idx].offload()
                if idx + 1 < len(past_key_values.layers):
                    past_key_values.layers[idx + 1].prefetch()

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3ForSequenceClassification(GenericForSequenceClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForTokenClassification(GenericForTokenClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForQuestionAnswering(GenericForQuestionAnswering, Qwen3PreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]
