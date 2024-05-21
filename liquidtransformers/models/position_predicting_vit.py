
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from transformers import ViTForImageClassification,ViTPreTrainedModel,ViTConfig
from transformers.models.vit.modeling_vit import ViTPooler,ViTEmbeddings,ViTLayer
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

@dataclass
class PositionAwareModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    position_logits: Optional[torch.FloatTensor] = None
    position_targets: Optional[torch.FloatTensor] = None

@dataclass
class PositionAwareModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    position_logits: Optional[torch.FloatTensor] = None
    position_targets: Optional[torch.FloatTensor] = None

class PositionMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class VitPositionAwareLayer(ViTLayer):
    def __init__(self, config):
        super().__init__(config)
        self.position_mlp = PositionMLP(config.hidden_size,config.num_hidden_layers)

    def forward(self, hidden_states, head_mask, output_attentions):
        output = super().forward(hidden_states, head_mask, output_attentions)
        pos_logits = self.position_mlp(output[0].mean(dim = 1))
        return output, pos_logits

class PositionPredictingViTEncoder(torch.nn.Module):
    def __init__(self, config, shuffle=True):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VitPositionAwareLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
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
        all_pos_logits = ()
        all_pos_targets = ()

        pos_loss = None
        self.pos_loss = None

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
                layer_outputs, pos_logits = layer_module(hidden_states, layer_head_mask, output_attentions)
                target_position_vec = torch.tensor([i] * hidden_states.shape[0], device=hidden_states.device)
                all_pos_logits = all_pos_logits + (pos_logits,)
                all_pos_targets = all_pos_targets + (target_position_vec,)


            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return PositionAwareModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            position_logits=all_pos_logits,
            position_targets=all_pos_targets
        )
class PositionPredictingViTModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = PositionPredictingViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return PositionAwareModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            position_logits=encoder_outputs.position_logits,
            position_targets=encoder_outputs.position_targets
        )

class PositionPredictingViTForImageClassification(ViTForImageClassification):

    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = PositionPredictingViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        #TODO: addition to loss goes here
        position_logits = torch.cat(outputs.position_logits)
        position_targets = torch.cat(outputs.position_targets)
        loss += cross_entropy(position_logits, position_targets)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )