"""
"Prunable" subclasses of HuggingFace transformers
    Classes that end with "Prunable" are the subclasses which support block-wise pruning.
    For some models such as MobileViT v1, the semantic/interface of attention-head-wise pruning is also improved.

model.prune_blocks(blocks_to_prune=pruning_idx)
    pruning_idx is a list of layer indices to prune (`int`, index begins from 0)
"""

import math
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from typing import Dict, List, Optional, Set, Tuple, Union

from transformers import ViTForImageClassification, ViTModel, ViTConfig
from transformers.models.mobilevit.configuration_mobilevit import MobileViTConfig
from transformers.models.vit.modeling_vit import ViTEncoder, ViTEmbeddings, ViTPooler

from transformers import MobileViTForImageClassification, MobileViTModel, MobileViTConfig
from transformers.models.mobilevit.modeling_mobilevit import MobileViTConvLayer, MobileViTEncoder
from transformers.models.mobilevit.modeling_mobilevit import MobileViTLayer, MobileViTMobileNetLayer
from transformers.models.mobilevit.modeling_mobilevit import MobileViTInvertedResidual, MobileViTTransformer
from transformers.models.mobilevit.modeling_mobilevit import MobileViTTransformerLayer, MobileViTOutput
from transformers.models.mobilevit.modeling_mobilevit import MobileViTAttention, MobileViTIntermediate
from transformers.models.mobilevit.modeling_mobilevit import MobileViTSelfAttention, MobileViTSelfOutput

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_outputs import ImageClassifierOutput, ImageClassifierOutputWithNoAttention
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention
from transformers.modeling_outputs import BaseModelOutputWithNoAttention

from mixup_utils import mixup_data, mixup_criterion, cutmixup_data, cutmix_data


class IdentityBlock(nn.Module):
    def __init__(self):
        super().__init__()


class IdentityViTLayer(IdentityBlock):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


class IdentityMobileViTTransformerLayer(IdentityBlock):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class IdentityMobileViTInvertedResidual(IdentityBlock):
    def __init__(self):
        super().__init__()

    def forward(self, features, *args, **kwargs):
        return features


class ViTEncoderPrunable(ViTEncoder):
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        alphas: torch.Tensor = None
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            alpha_ = alphas[i] if alphas is not None else None
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, layer_head_mask, output_attentions)

            if alpha_ is not None:
                hidden_states = layer_outputs[0] * \
                    alpha_ + hidden_states * (1 - alpha_)
            else:
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


class ViTModelPrunable(ViTModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        # intentionally skip the __init__ of ViTModel
        super(ViTModel, self).__init__(config)

        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoderPrunable(config)

        self.layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def prune_blocks(self, blocks_to_prune):
        for block_idx in blocks_to_prune:
            self.encoder.layer[block_idx] = IdentityViTLayer()

    def forward_classification_feature(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        alphas: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # see ViTForImageClassification

        # do not directly call .forward(), call .__call__() instead
        outputs = self(pixel_values=pixel_values,
                       head_mask=head_mask,
                       alphas=alphas,
                       bool_masked_pos=bool_masked_pos,
                       output_attentions=output_attentions,
                       output_hidden_states=output_hidden_states,
                       interpolate_pos_encoding=interpolate_pos_encoding,
                       return_dict=return_dict)

        sequence_output = outputs[0]
        classification_feature = sequence_output[:, 0, :]

        return classification_feature

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        alphas: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
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
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

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
            alphas=alphas
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (
                sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTForImageClassificationPrunable(ViTForImageClassification):
    def __init__(self, config: ViTConfig) -> None:
        # intentionally skip the __init__ of ViTForImageClassification
        super(ViTForImageClassification, self).__init__(config)

        self.num_labels = config.num_labels
        self.vit = ViTModelPrunable(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(
            config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def prune_blocks(self, blocks_to_prune):
        # self.base_model == self.vit
        self.vit.prune_blocks(blocks_to_prune)

    def forward_classification_feature(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        alphas: Optional[torch.Tensor] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        return self.vit.forward_classification_feature(pixel_values=pixel_values,
                                                       head_mask=head_mask,
                                                       alphas=alphas,
                                                       output_attentions=output_attentions,
                                                       output_hidden_states=output_hidden_states,
                                                       interpolate_pos_encoding=interpolate_pos_encoding,
                                                       return_dict=return_dict)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        alphas: Optional[torch.Tensor] = None,
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
            alphas=alphas,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
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
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MobileViTModelPrunable(MobileViTModel):
    """
    MobileViT v1 small (apple/mobilevit-small) has 30 blocks/layers in total. The definition of 
        a block/layer is from the figure of the original paper. Out of these 30 blocks/layers, there 
        are 14 usual convolution which do not have skip-connections, there are 9 transformer blocks 
        which all have skip-connections, there are 7 MobileNetV2 convolution blocks. Only 2 out of 
        these 7 MobileNetV2 blocks have skip-connections. So, out of these 30 blocks/layers, only 
        9 transformer blocks and 2 MobileNetV2 convolution blocks have skip-connections, the other 
        19 blocks/layers do not have skip-connections. 

    MobileViTConvLayer (nn.Conv2d + nn.BatchNorm2d + activation, no skip-connection): 14
    MobileViTInvertedResidual (expand_1x1, conv_3x3, reduce_1x1. May have skip-connection: use_residual): 7
        2 out of the 7 MobileViTInvertedResidual have a skip-connection (the third one and the fourth one)
        5 out of the 7 MobileViTInvertedResidual do not have a skip-connection
    MobileViTTransformerLayer (2 + 4 + 3. All have skip-connection): 9
    
    prune_blocks allows pruning the 9 transformer blocks (indices from 0 to 8) but also 
        the 2 MobileNetV2 convolution blocks which have skip-connections (indices 9 and 10).
    
    The HF's default prune_heads has 2 drawbacks:
        1. its block/layer indexing is misleading
        2. it forces pruning the same heads for each of the 3 phases of transformer blocks
        
    self.encoder.layer[0]: MobileViTMobileNetLayer(stride=1, num_stages=1)
    self.encoder.layer[1]: MobileViTMobileNetLayer(stride=2, num_stages=3)
    self.encoder.layer[2]: MobileViTLayer(stride=2, num_stages=2)
    self.encoder.layer[3]: MobileViTLayer(stride=2, num_stages=4)
    self.encoder.layer[4]: MobileViTLayer(stride=2, num_stages=3)
    """

    def __init__(self, config: MobileViTConfig, expand_output: bool = True):
        # intentionally skip the __init__ of MobileViTModel
        super(MobileViTModel, self).__init__(config)

        self.config = config
        self.expand_output = expand_output

        self.conv_stem = MobileViTConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=config.neck_hidden_sizes[0],
            kernel_size=3,
            stride=2,
        )

        self.encoder = MobileViTEncoderPrunable(config)

        if self.expand_output:
            self.conv_1x1_exp = MobileViTConvLayer(
                config,
                in_channels=config.neck_hidden_sizes[5],
                out_channels=config.neck_hidden_sizes[6],
                kernel_size=1,
            )

        # Initialize weights and apply final processing
        self.post_init()

        # For MobileViT v1, the number of transformer
        # blocks is hard-coded and equals (2 + 4 + 3).
        self.num_transformer_blocks = 2 + 4 + 3

        # plus 2 MobileNetV2 convolution blocks which have skip-connections
        self.num_prunable_blocks = self.num_transformer_blocks + 2

    def prune_blocks(self, blocks_to_prune):
        """
        indexing of blocks_to_prune:
            starting from 0 to 10 (11 prunable blocks in total)
            Transformer blocks are represented by {0, 1, 2, ..., 8}, which mean the first 9 indices.
            The 2 MobileNetV2 convolution blocks are represented by the indices 9 and 10.
        """
        for index_ in blocks_to_prune:
            if index_ in [0, 1]:
                transformed_idx = index_
                self.encoder.layer[2].transformer.layer[transformed_idx] = IdentityMobileViTTransformerLayer(
                )
            elif index_ in [2, 3, 4, 5]:
                transformed_idx = index_ - 2
                self.encoder.layer[3].transformer.layer[transformed_idx] = IdentityMobileViTTransformerLayer(
                )
            elif index_ in [6, 7, 8]:
                transformed_idx = index_ - 6
                self.encoder.layer[4].transformer.layer[transformed_idx] = IdentityMobileViTTransformerLayer(
                )
            elif index_ in [9, 10]:
                # +1 because the first MobileNetV2 convolution block is for the downsampling
                transformed_idx = index_ - 9 + 1
                self.encoder.layer[1].layer[transformed_idx] = IdentityMobileViTInvertedResidual(
                )
            else:
                raise Exception(
                    "Block index {} is out of bounds.".format(index_))

    def _prune_heads(self, heads_to_prune):
        """
        Rewrite _prune_heads to improve it.
        
        The keys of heads_to_prune (dict) must be integers from 0 to 8 (both inclusive).
        """
        for layer_index, heads in heads_to_prune.items():
            if layer_index in [0, 1]:
                transformed_idx = layer_index
                self.encoder.layer[2].transformer.layer[transformed_idx].attention.prune_heads(
                    heads)
            elif layer_index in [2, 3, 4, 5]:
                transformed_idx = layer_index - 2
                self.encoder.layer[3].transformer.layer[transformed_idx].attention.prune_heads(
                    heads)
            elif layer_index in [6, 7, 8]:
                transformed_idx = layer_index - 6
                self.encoder.layer[4].transformer.layer[transformed_idx].attention.prune_heads(
                    heads)
            elif layer_index in [9, 10]:
                raise Exception(
                    "Block {} is not a transformer block.".format(layer_index))
            else:
                raise Exception(
                    "Block index {} is out of bounds.".format(layer_index))

    def forward_classification_feature(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head_mask: Optional[torch.Tensor] = None,
        alphas: Optional[torch.Tensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        # see MobileViTForImageClassification

        # do not directly call .forward(), call .__call__() instead
        outputs = self(pixel_values=pixel_values,
                       head_mask=head_mask,
                       alphas=alphas,
                       output_hidden_states=output_hidden_states,
                       return_dict=return_dict)
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        classification_feature = pooled_output
        return classification_feature

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head_mask: Optional[torch.Tensor] = None,
        alphas: Optional[torch.Tensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # the following get_head_mask is our modification (imitating ViTModel)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.num_transformer_blocks)

        embedding_output = self.conv_stem(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_mask=head_mask,
            alphas=alphas,
        )

        if self.expand_output:
            last_hidden_state = self.conv_1x1_exp(encoder_outputs[0])

            # global average pooling: (batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = torch.mean(
                last_hidden_state, dim=[-2, -1], keepdim=False)
        else:
            last_hidden_state = encoder_outputs[0]
            pooled_output = None

        if not return_dict:
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (
                last_hidden_state,)
            return output + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class MobileViTForImageClassificationPrunable(MobileViTForImageClassification):
    def __init__(self, config: MobileViTConfig) -> None:
        # intentionally skip the __init__ of MobileViTForImageClassification
        super(MobileViTForImageClassification, self).__init__(config)

        self.num_labels = config.num_labels
        self.mobilevit = MobileViTModelPrunable(config)

        # Classifier head
        self.dropout = nn.Dropout(config.classifier_dropout_prob, inplace=True)
        self.classifier = (
            nn.Linear(
                config.neck_hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

        # For MobileViT v1, the number of transformer
        # blocks is hard-coded and equals (2 + 4 + 3).
        self.num_transformer_blocks = 2 + 4 + 3

        # plus 2 MobileNetV2 convolution blocks which have skip-connections
        self.num_prunable_blocks = self.num_transformer_blocks + 2

    def prune_blocks(self, blocks_to_prune):
        # self.base_model == self.mobilevit
        self.mobilevit.prune_blocks(blocks_to_prune)

    def forward_classification_feature(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        head_mask: Optional[torch.Tensor] = None,
        alphas: Optional[torch.Tensor] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        return self.mobilevit.forward_classification_feature(pixel_values=pixel_values,
                                                             head_mask=head_mask,
                                                             alphas=alphas,
                                                             output_hidden_states=output_hidden_states,
                                                             return_dict=return_dict)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        head_mask: Optional[torch.Tensor] = None,
        alphas: Optional[torch.Tensor] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilevit(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict,
            head_mask=head_mask, alphas=alphas)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
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
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class MobileViTForImageClassificationPrunableCutmix(MobileViTForImageClassificationPrunable):
    """
    A subclass of MobileViTForImageClassificationPrunable which can uses the cutmix 
    data augmentation (for ImageNet fine-tuning)
    """
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        head_mask: Optional[torch.Tensor] = None,
        alphas: Optional[torch.Tensor] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pixel_values, targets_a, targets_b, lam = cutmix_data(
            pixel_values, labels, alpha=0.2)

        outputs = self.mobilevit(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict,
            head_mask=head_mask, alphas=alphas)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
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
                criterion = CrossEntropyLoss()
                # loss = loss_fct(
                #     logits.view(-1, self.num_labels), labels.view(-1))
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(criterion, logits)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def forward_test(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        head_mask: Optional[torch.Tensor] = None,
        alphas: Optional[torch.Tensor] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilevit(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict,
            head_mask=head_mask, alphas=alphas)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
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
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class MobileViTEncoderPrunable(MobileViTEncoder):
    def __init__(self, config: MobileViTConfig) -> None:
        # intentionally skip the __init__ of MobileViTEncoder
        super(MobileViTEncoder, self).__init__()

        self.config = config

        self.layer = nn.ModuleList()
        self.gradient_checkpointing = False

        # segmentation architectures like DeepLab and PSPNet modify the strides
        # of the classification backbones
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        elif config.output_stride == 16:
            dilate_layer_5 = True

        dilation = 1

        layer_1 = MobileViTMobileNetLayerPrunable(
            config,
            in_channels=config.neck_hidden_sizes[0],
            out_channels=config.neck_hidden_sizes[1],
            stride=1,
            num_stages=1,
        )
        self.layer.append(layer_1)

        layer_2 = MobileViTMobileNetLayerPrunable(
            config,
            in_channels=config.neck_hidden_sizes[1],
            out_channels=config.neck_hidden_sizes[2],
            stride=2,
            num_stages=3,
        )
        self.layer.append(layer_2)

        layer_3 = MobileViTLayerPrunable(
            config,
            in_channels=config.neck_hidden_sizes[2],
            out_channels=config.neck_hidden_sizes[3],
            stride=2,
            hidden_size=config.hidden_sizes[0],
            num_stages=2,
        )
        self.layer.append(layer_3)

        if dilate_layer_4:
            dilation *= 2

        layer_4 = MobileViTLayerPrunable(
            config,
            in_channels=config.neck_hidden_sizes[3],
            out_channels=config.neck_hidden_sizes[4],
            stride=2,
            hidden_size=config.hidden_sizes[1],
            num_stages=4,
            dilation=dilation,
        )
        self.layer.append(layer_4)

        if dilate_layer_5:
            dilation *= 2

        layer_5 = MobileViTLayerPrunable(
            config,
            in_channels=config.neck_hidden_sizes[4],
            out_channels=config.neck_hidden_sizes[5],
            stride=2,
            hidden_size=config.hidden_sizes[2],
            num_stages=3,
            dilation=dilation,
        )
        self.layer.append(layer_5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        head_mask: Optional[torch.Tensor] = None,
        alphas: Optional[torch.Tensor] = None,
    ) -> Union[tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        curr_head_mask = None
        curr_alphas = None
        for i, layer_module in enumerate(self.layer):
            if i == 2:
                # first MobileViTLayerPrunable
                curr_head_mask = head_mask[:2] if head_mask is not None else None
                curr_alphas = alphas[:2] if alphas is not None else None
            elif i == 3:
                # second MobileViTLayerPrunable
                curr_head_mask = head_mask[2:6] if head_mask is not None else None
                curr_alphas = alphas[2:6] if alphas is not None else None
            elif i == 4:
                # third MobileViTLayerPrunable
                curr_head_mask = head_mask[6:9] if head_mask is not None else None
                curr_alphas = alphas[6:9] if alphas is not None else None

            if self.gradient_checkpointing and self.training:
                # TODO: this part does not support head_mask and alphas
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                hidden_states = layer_module(
                    hidden_states, head_mask=curr_head_mask, alphas=curr_alphas)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class MobileViTMobileNetLayerPrunable(MobileViTMobileNetLayer):
    def forward(self, features: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None,
                alphas: Optional[torch.Tensor] = None,) -> torch.Tensor:
        for layer_module in self.layer:
            features = layer_module(features)
        return features


class MobileViTLayerPrunable(MobileViTLayer):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        hidden_size: int,
        num_stages: int,
        dilation: int = 1,
    ) -> None:
        # intentionally skip the __init__ of MobileViTLayer
        super(MobileViTLayer, self).__init__()

        self.patch_width = config.patch_size
        self.patch_height = config.patch_size

        if stride == 2:
            self.downsampling_layer = MobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
            )
            in_channels = out_channels
        else:
            self.downsampling_layer = None

        self.conv_kxk = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
        )

        self.conv_1x1 = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        self.transformer = MobileViTTransformerPrunable(
            config,
            hidden_size=hidden_size,
            num_stages=num_stages,
        )

        self.layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        self.conv_projection = MobileViTConvLayer(
            config, in_channels=hidden_size, out_channels=in_channels, kernel_size=1
        )

        self.fusion = MobileViTConvLayer(
            config, in_channels=2 * in_channels, out_channels=in_channels, kernel_size=config.conv_kernel_size
        )

    def forward(self, features: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None,
                alphas: Optional[torch.Tensor] = None,) -> torch.Tensor:
        # reduce spatial dimensions if needed
        if self.downsampling_layer:
            features = self.downsampling_layer(features)

        residual = features

        # local representation
        features = self.conv_kxk(features)
        features = self.conv_1x1(features)

        # convert feature map to patches
        patches, info_dict = self.unfolding(features)

        # learn global representations
        patches = self.transformer(patches, head_mask=head_mask, alphas=alphas)
        patches = self.layernorm(patches)

        # convert patches back to feature maps
        features = self.folding(patches, info_dict)

        features = self.conv_projection(features)
        features = self.fusion(torch.cat((residual, features), dim=1))
        return features


class MobileViTTransformerPrunable(MobileViTTransformer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, num_stages: int) -> None:
        # intentionally skip the __init__ of MobileViTTransformer
        super(MobileViTTransformer, self).__init__()

        self.layer = nn.ModuleList()
        for _ in range(num_stages):
            transformer_layer = MobileViTTransformerLayerPrunable(
                config,
                hidden_size=hidden_size,
                intermediate_size=int(hidden_size * config.mlp_ratio),
            )
            self.layer.append(transformer_layer)

    def forward(self, hidden_states: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None,
                alphas: Optional[torch.Tensor] = None,) -> torch.Tensor:
        for i, layer_module in enumerate(self.layer):

            alpha_ = alphas[i] if alphas is not None else None
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states, head_mask=layer_head_mask)

            if alpha_ is not None:
                hidden_states = layer_outputs * \
                    alpha_ + hidden_states * (1 - alpha_)
            else:
                hidden_states = layer_outputs

        return hidden_states


class MobileViTTransformerLayerPrunable(MobileViTTransformerLayer):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int) -> None:
        # intentionally skip the __init__ of MobileViTTransformerLayer
        super(MobileViTTransformerLayer, self).__init__()

        self.attention = MobileViTAttentionPrunable(config, hidden_size)
        self.intermediate = MobileViTIntermediate(
            config, hidden_size, intermediate_size)
        self.output = MobileViTOutput(config, hidden_size, intermediate_size)
        self.layernorm_before = nn.LayerNorm(
            hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(
            hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attention_output = self.attention(
            self.layernorm_before(hidden_states), head_mask=head_mask)
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return layer_output


class MobileViTAttentionPrunable(MobileViTAttention):
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        # intentionally skip the __init__ of MobileViTAttention
        super(MobileViTAttention, self).__init__()

        self.attention = MobileViTSelfAttentionPrunable(config, hidden_size)
        self.output = MobileViTSelfOutput(config, hidden_size)
        self.pruned_heads = set()

    def forward(self, hidden_states: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self_outputs = self.attention(hidden_states, head_mask=head_mask)
        attention_output = self.output(self_outputs)
        return attention_output


class MobileViTSelfAttentionPrunable(MobileViTSelfAttention):
    def forward(self, hidden_states: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # the following masking is our modification (compared to MobileViTSelfAttention),
        # this modification imitates ViTSelfAttention of HF's modeling_vit.py

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
