import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Union
from torch.nn import CrossEntropyLoss
from transformers.generation import GenerationMixin
from transformers.utils import logging


logger = logging.get_logger(__name__)


class CustomSplitLLamaModel(LlamaModel):
    config_class = LlamaConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        del self.layers

        config.path8b = "/fsxp3/meta-llama_Llama-3.1-8B-Instruct"
        config.path70b = "/fsxp3/meta-llama_Llama-3.3-70B-Instruct"

        # Load configs and define layer counts
        config_8b = LlamaConfig.from_pretrained(config.path8b)
        config_70b = LlamaConfig.from_pretrained(config.path70b)

        config_8b._attn_implementation = "flash_attention_2"
        config_70b._attn_implementation = "flash_attention_2"

        self.mlp = config.mlp if hasattr(config, "mlp") else False
        self.num_layers_8b = config.num_layers_8
        self.num_layers_70b = config.num_layers_70

        # Build the model architecture
        self.embed_tokens = nn.Embedding(config.vocab_size, config_8b.hidden_size, self.padding_idx)

        # First set of layers (from 8B config) - LlamaDecoderLayer handles checkpointing
        self.layers_first = nn.ModuleList(
            [LlamaDecoderLayer(config_8b, layer_idx=i) for i in range(self.num_layers_8b)]
        )

        # Adapter to bridge the hidden dimensions
        if self.mlp:
            self.adapter_linear_1 = nn.Linear(config_8b.hidden_size, config_70b.hidden_size, bias=False)
            self.adapter_linear_2 = nn.Linear(config_70b.hidden_size, config_70b.hidden_size, bias=False)
        else:
            self.adapter = nn.Linear(config_8b.hidden_size, config_70b.hidden_size, bias=False)

        # Last set of layers (from 70B config) - LlamaDecoderLayer handles checkpointing
        start_idx_70b = config_70b.num_hidden_layers - self.num_layers_70b
        self.layers_last = nn.ModuleList(
            [
                LlamaDecoderLayer(config_70b, layer_idx=self.num_layers_8b + i - start_idx_70b)
                for i in range(start_idx_70b, config_70b.num_hidden_layers)
            ]
        )

        self.norm = LlamaRMSNorm(config_70b.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config_70b.hidden_size, config.vocab_size, bias=False)

        self.rotary_emb_8b = LlamaRotaryEmbedding(config_8b)
        self.rotary_emb_70b = LlamaRotaryEmbedding(config_70b)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Initialize cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # Get past length
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        # Handle cache position
        if cache_position is None:
            cache_position = torch.arange(
                past_length,
                past_length + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Handle position IDs
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Handle attention mask for cached generation
        if use_cache and past_length > 0:
            if attention_mask is not None and attention_mask.shape[1] != past_length + inputs_embeds.shape[1]:
                # Extend attention mask to cover cached tokens
                batch_size = inputs_embeds.shape[0]
                attention_mask = torch.ones(
                    (batch_size, past_length + inputs_embeds.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Compute position embeddings for 8B layers
        position_embeddings_8b = self.rotary_emb_8b(hidden_states, position_ids)

        # Process 8B layers - gradient checkpointing handled by GradientCheckpointingLayer
        for decoder_layer in self.layers_first:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings_8b,
            )

            if output_attentions:
                all_self_attns += (hidden_states.attentions if hasattr(hidden_states, "attentions") else None,)

        # Apply adapter
        if self.mlp:
            hidden_states = torch.relu(self.adapter_linear_1(hidden_states))
            hidden_states = self.adapter_linear_2(hidden_states)
        else:
            hidden_states = self.adapter(hidden_states)

        # Compute position embeddings for 70B layers
        position_embeddings_70b = self.rotary_emb_70b(hidden_states, position_ids)

        # Process 70B layers - gradient checkpointing handled by GradientCheckpointingLayer
        for decoder_layer in self.layers_last:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings_70b,
            )

            if output_attentions:
                all_self_attns += (hidden_states.attentions if hasattr(hidden_states, "attentions") else None,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CustomSplitLLamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomSplitLLamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = self.model.lm_head
        del self.model.lm_head

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits
        if num_logits_to_keep > 0:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        else:
            logits = self.lm_head(hidden_states)

        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Slice input_ids for cached generation
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        # Create position_ids from attention_mask
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # Extend attention mask for cached generation
        if past_key_values is not None and attention_mask is not None:
            past_length = past_key_values.get_seq_length()
            if attention_mask.shape[1] < past_length + input_ids.shape[1]:
                batch_size = input_ids.shape[0]
                attention_mask = torch.ones(
                    (batch_size, past_length + input_ids.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

        # Use inputs_embeds only in first generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
