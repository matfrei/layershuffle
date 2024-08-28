import torch
import torch.nn as nn

from transformers import ViTForImageClassification,ViTModel,ViTConfig
from transformers.models.vit.modeling_vit import ViTPooler,ViTEmbeddings,ViTLayer

from transformers.models.vit.modeling_vit import ViTEncoder,ViTPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
# FIXXXME: this approach is awfully verbose and duplicates a lot of code
# there has to be a better way that is clean?

class ShufflingViTEncoder(ViTEncoder):
    def __init__(self, config, shuffle=True):
        super().__init__(config)
        self.shuffle = shuffle

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if self.shuffle:
            idx_order = torch.randperm(len(self.layer)).to(hidden_states.device)
        else:
            idx_order = range(len(self.layer))

        for i, permuted_idx in enumerate(idx_order):
            layer_module = self.layer[permuted_idx]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class ShufflingViTModel(ViTModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config,add_pooling_layer,use_mask_token)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ShufflingViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

class ShufflingViTForImageClassification(ViTForImageClassification):

    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = ShufflingViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()


