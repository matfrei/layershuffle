import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput
from einops import repeat

from transformers import ViTForImageClassification,ViTModel,ViTConfig
from transformers.models.vit.modeling_vit import ViTPooler,ViTEmbeddings,ViTLayer

class PositionProjection(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)
class VitProjectedLayer(ViTLayer):
    def __init__(self, config):
        super().__init__(config)
        # ok, here we have to find out what dim is
        self.projection_dim = 32
        self.projection = PositionProjection(config.hidden_size + self.projection_dim, config.hidden_size, dropout=0)

    def forward(self, hidden_states, head_mask, output_attentions, depth_embedding):
        hidden_states_cat = torch.cat((hidden_states, depth_embedding), dim=-1)
        hidden_states = self.projection(hidden_states_cat) + hidden_states
        return super().forward(hidden_states, head_mask, output_attentions)

class PositionEncodingViTEncoder(torch.nn.Module):
    def __init__(self, config, shuffle=True):
        super().__init__()

        self.pos_enc_dim = 32

        self.config = config
        self.layer = nn.ModuleList([VitProjectedLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.embedding = torch.nn.Embedding(config.num_hidden_layers, self.pos_enc_dim)
        torch.nn.init.zeros_(self.embedding.weight)
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
        b, n, _ = hidden_states.shape

        if self.shuffle:
            idx_order = torch.randperm(len(self.layer)).to(hidden_states.device)
        else:
            idx_order = range(len(self.layer))

        for i, permuted_idx in enumerate(idx_order):
            depth_embedding = self.embedding(torch.LongTensor([i]).to(hidden_states.device))
            depth_embedding = repeat(depth_embedding, '1 d -> b n d', b=b, n=n)

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
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, depth_embedding)

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

class PositionEncodingViTModel(ViTModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = PositionEncodingViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

class PositionEncodingViTForImageClassification(ViTForImageClassification):

    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = PositionEncodingViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()
