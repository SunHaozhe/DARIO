import logging
import torch.nn as nn
from transformers import ViTImageProcessor
from transformers import MobileViTImageProcessor

from prunable_model_classes import ViTForImageClassificationPrunable, ViTModelPrunable
from prunable_model_classes import MobileViTForImageClassificationPrunable, MobileViTModelPrunable

from transformers.models.vit.modeling_vit import ViTLayer
from prunable_model_classes import MobileViTTransformerLayerPrunable

from prunable_model_classes import MobileViTForImageClassificationPrunableCutmix


# first element: class with classification head, 
# second element: class without classification head, 
# third element: checkpoint name
_model_class_register_table = {
    "MAE_ViT_base": [ViTForImageClassificationPrunable, 
                     ViTModelPrunable, 
                     "facebook/vit-mae-base"],
    "MobileViT_small": [MobileViTForImageClassificationPrunable, 
                        MobileViTModelPrunable, 
                        "apple/mobilevit-small"],
}


def _nn_model_lookup_function(input_name):
    """
    converts input model name to checkpoint name on the Hub (public or private) and its 
    corresponding classes.
    
    we define a function instead of a dictionary in order to handle our own private checkpoints
    
    input_name: str (two possible types)
        1. HzsAlhTransformerProject/XXX                   (our own checkpoints)
        2. keys from _model_class_register_table.keys()   (public hub checkpoints)
    """
    class_with_chead, class_without_chead, checkpoint_name = None, None, None
    
    if input_name.startswith("HzsAlhTransformerProject/"):
        # our own checkpoints
        checkpoint_name = input_name
        
        hub_organization_name, complete_model_name = input_name.split("/")
        for k in _model_class_register_table.keys():
            if complete_model_name.startswith(k):
                class_with_chead, class_without_chead, _ = _model_class_register_table[k]
                break
    else:
        for k in _model_class_register_table.keys():
            if input_name == k:
                class_with_chead, class_without_chead, checkpoint_name = _model_class_register_table[k]
                break
        
    assert (class_with_chead is not None) and (class_without_chead is not None) and (
        checkpoint_name is not None), "Model {} not recognized.".format(input_name)
    
    return checkpoint_name, class_with_chead, class_without_chead


def get_pretrained_preprocessor(args):
    checkpoint_name, _, _ = _nn_model_lookup_function(
        args.model)
    
    if args.model == "MAE_ViT_base":
        image_processor = ViTImageProcessor.from_pretrained(checkpoint_name)
    elif args.model == "MobileViT_small":
        # Default MobileViTImageProcessor (transformers v4.27.4) first 
        # resizes to {"shortest_edge": 288}, then center-crops to 
        # {"height": 256, "width": 256}.
        # To avoid potential loss of information on our data due to cropping, we 
        # resize to {"shortest_edge": 256}, then center-crop to
        # {"height": 256, "width": 256}.
        image_processor = MobileViTImageProcessor.from_pretrained(
            checkpoint_name, size={"shortest_edge": 256}, do_center_crop=True,
            crop_size={"height": 256, "width": 256})
    else:
        raise Exception(
            "Pre-trained preprocessor {} not recognized.".format(args.model))
    
    return image_processor


def _get_model_with_classification_head_or_not(checkpoint_name, class_with_chead, 
                                               class_without_chead, 
                                               with_classification_head, 
                                               do_post_processing, 
                                               num_classes=None):
    if with_classification_head:
        # ignore_mismatched_sizes=True because some pre-trained models already 
        # has a classification head which has a different size
        model = class_with_chead.from_pretrained(
            checkpoint_name, num_labels=num_classes, ignore_mismatched_sizes=True)
        
        if do_post_processing:
            # Re-initializes the classification head.
            # This makes sure of not beginning from the pre-trained weights of the classifier 
            # but this step may not be necessary because maybe it is already handled 
            # by .from_pretrained().
            assert hasattr(model, "classifier")
            classifier_in_features = model.classifier.in_features
            model.classifier = nn.Linear(
                in_features=classifier_in_features, out_features=num_classes, bias=True)
            
            freeze_model_parameters(
                model, how="freeze_all_except_classification_head")
    else:
        model = class_without_chead.from_pretrained(checkpoint_name)
        
        if do_post_processing:
            # Freezes all parameters
            freeze_model_parameters(model, how="freeze_all")
    return model


def freeze_model_parameters(model, how):
    if how == "freeze_all_except_classification_head":
        # Freezes all parameters except for the classification head.
        # We can probably use model.base_model to accelerate a little bit
        # if the search speed is really critical.
        for name, param in model.named_parameters():
            if name.startswith("classifier"):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif how == "freeze_all":
        for name, param in model.named_parameters():
            param.requires_grad = False
    elif how == "unfreeze_all":
        for name, param in model.named_parameters():
            param.requires_grad = True
    elif how == "unfreeze_last_block":
        if isinstance(model, ViTForImageClassificationPrunable):
            # find real blocks (excluding placeholder empty blocks)
            indices_real_blocks = []
            for idx, block_ in enumerate(model.vit.encoder.layer):
                if isinstance(block_, ViTLayer):
                    indices_real_blocks.append(idx)
                    
            # select the blocks to unfreeze
            indices_real_blocks_to_unfreeze = indices_real_blocks[-1]
            if not isinstance(indices_real_blocks_to_unfreeze, list):
                indices_real_blocks_to_unfreeze = [indices_real_blocks_to_unfreeze]
            
            # build the list of parameter name prefixes
            
            # We don't contain vit.pooler because ViTForImageClassification 
            # does not have a pooler
            param_name_prefix_to_unfreeze = ["classifier", "vit.layernorm"]
            for idx in indices_real_blocks_to_unfreeze:
                param_name_prefix_to_unfreeze.append("vit.encoder.layer.{}".format(idx))
            param_name_prefix_to_unfreeze = tuple(param_name_prefix_to_unfreeze)
            
            # do the real job
            for name, param in model.named_parameters():
                if name.startswith(param_name_prefix_to_unfreeze):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif isinstance(model, MobileViTForImageClassificationPrunable):
            # collect all ViT blocks into one place
            all_ViT_blocks_in_MobileViT = []
            for idx, block_ in enumerate(model.mobilevit.encoder.layer[2].transformer.layer):
                all_ViT_blocks_in_MobileViT.append(
                    (block_, "mobilevit.encoder.layer.2.transformer.layer.{}".format(idx))
                )
            for idx, block_ in enumerate(model.mobilevit.encoder.layer[3].transformer.layer):
                all_ViT_blocks_in_MobileViT.append(
                    (block_, "mobilevit.encoder.layer.3.transformer.layer.{}".format(idx))
                )
            for idx, block_ in enumerate(model.mobilevit.encoder.layer[4].transformer.layer):
                all_ViT_blocks_in_MobileViT.append(
                    (block_, "mobilevit.encoder.layer.4.transformer.layer.{}".format(idx))
                )
            
            # find real blocks (excluding placeholder empty blocks)
            indices_real_blocks = []
            for idx, (block_, param_name_prefix) in enumerate(all_ViT_blocks_in_MobileViT):
                if isinstance(block_, MobileViTTransformerLayerPrunable):
                    indices_real_blocks.append(idx)
            
            # select the blocks to unfreeze
            indices_real_blocks_to_unfreeze = indices_real_blocks[-1]
            if not isinstance(indices_real_blocks_to_unfreeze, list):
                indices_real_blocks_to_unfreeze = [indices_real_blocks_to_unfreeze]
            
            # build the list of parameter name prefixes
            param_name_prefix_to_unfreeze = ["classifier", "mobilevit.conv_1x1_exp"]
            for idx in indices_real_blocks_to_unfreeze:
                block_, param_name_prefix = all_ViT_blocks_in_MobileViT[idx]
                param_name_prefix_to_unfreeze.append(param_name_prefix)
            
            remaining_prefix_to_unfreeze = []
            for xx in param_name_prefix_to_unfreeze:
                for outer_idx in [2, 3, 4]:
                    if "mobilevit.encoder.layer.{}.transformer.layer".format(outer_idx) in xx:
                        # the last part of the current MobileViTLayer
                        remaining_prefix_to_unfreeze.append(
                            "mobilevit.encoder.layer.{}.layernorm".format(outer_idx))
                        remaining_prefix_to_unfreeze.append(
                            "mobilevit.encoder.layer.{}.conv_projection".format(outer_idx))
                        remaining_prefix_to_unfreeze.append(
                            "mobilevit.encoder.layer.{}.fusion".format(outer_idx))
                        # the first part of the next MobileViTLayer
                        if outer_idx in [2, 3]:
                            remaining_prefix_to_unfreeze.append(
                                "mobilevit.encoder.layer.{}.downsampling_layer".format(outer_idx + 1))
                            remaining_prefix_to_unfreeze.append(
                                "mobilevit.encoder.layer.{}.conv_kxk".format(outer_idx + 1))
                            remaining_prefix_to_unfreeze.append(
                                "mobilevit.encoder.layer.{}.conv_1x1".format(outer_idx + 1))
                        
            remaining_prefix_to_unfreeze = sorted(list(set(remaining_prefix_to_unfreeze)))
            param_name_prefix_to_unfreeze = param_name_prefix_to_unfreeze + remaining_prefix_to_unfreeze
            
            param_name_prefix_to_unfreeze = tuple(param_name_prefix_to_unfreeze)
            
            # do the real job
            for name, param in model.named_parameters():
                if name.startswith(param_name_prefix_to_unfreeze):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise Exception(
                "Model {} does not support {}".format(type(model), how))
    else:
        raise Exception(
            "Unrecognized argument {} for freeze_model_parameters".format(how))


def get_pretrained_model(model_name, device="cpu", num_classes=None, 
                         with_classification_head=False, return_layer_head_count=False, 
                         ignore_logging=True, do_post_processing=True, dataset_name=None):
    """
    with_classification_head: bool
        whether the returned model has a classification head with num_classes classes. 
    ignore_logging: bool
        ignores the loading warnings raised by HuggingFace. This feature is useful 
            during the iterative search
    do_post_processing: bool
        Whether re-initializes the classification head (if any) and freezes the backbone 
            except for the classification head (if any).
        This may not be necessary but this feature is here to guarantee that these 
            are properly done. 
        Maybe set it to False when the model instantiation speed is critical e.g. during the iterative search. 
        
    dataset_name: str
        an ad-hoc optional argument to enable the usage of Cutmix data augmentation for ImageNet experiments
    """
    if with_classification_head:
        assert num_classes is not None, "with_classification_head={}, you must specify num_classes.".format(
            with_classification_head)
    
    checkpoint_name, class_with_chead, class_without_chead = _nn_model_lookup_function(model_name)
    
    # The following is ad-hoc (only for ImageNet experiments: Cutmix data augmentation). 
    # It may need to be improved for generalized usage 
    # (e.g. other pre-trained models, class without chead)
    if dataset_name is not None and dataset_name == "imagenet":
        # currently, only MobileViT supports ImageNet experiments
        if class_with_chead == MobileViTForImageClassificationPrunable:
            class_with_chead = MobileViTForImageClassificationPrunableCutmix
        else:
            raise NotImplementedError(
                "Currently {} does not support ImageNet experiments.".format(dataset_name))
    
    if ignore_logging:
        context_manager_class = DisableLoggerContextManager
    else:
        context_manager_class = DummnyContextManager
    
    with context_manager_class():
        if model_name == "MAE_ViT_base":
            
            model = _get_model_with_classification_head_or_not(checkpoint_name, class_with_chead,
                                                               class_without_chead,
                                                               with_classification_head,
                                                               do_post_processing, 
                                                               num_classes=num_classes)
            
            if return_layer_head_count:
                num_layers = model.config.num_hidden_layers
                num_heads = model.config.num_attention_heads
                
        elif model_name == "MobileViT_small":
            
            model = _get_model_with_classification_head_or_not(checkpoint_name, class_with_chead,
                                                               class_without_chead,
                                                               with_classification_head,
                                                               do_post_processing, 
                                                               num_classes=num_classes)
            
            if return_layer_head_count:
                num_layers = model.num_transformer_blocks
                num_heads = model.config.num_attention_heads
                
        else:
            raise Exception(
                "Pre-trained model {} not recognized.".format(model_name))
        
    model.to(device)
    
    if return_layer_head_count:
        return model, num_layers, num_heads
    else:
        return model


def sampling(model_name, granularity, pruning_idx, 
             device="cpu", num_classes=None,
             with_classification_head=False, 
             dataset_name=None):
    """
    creates a pruned model according to model_name, granularity, pruning_idx.
    This function is to be called at each search step. 
    
    pruning_idx (python dictionary) is the outcome of each search step. 
    The precise format of pruning_idx depends on the granularity. 
    
    If granularity == "attention_head", pruning_idx should be a dictionary with keys 
        being selected layer indices (`int`, index begins from 0) and associated values 
        being the list of heads to prune in said layer (list of `int`, index begins from 0). 
        For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on layer 1 and 
        heads 2 and 3 on layer 2.
    
    Possible granularity levels:
        1. One individual parameter (a scalar)
        2. One weight matrix Q, K, or V in a certain self-attention head 
        3. One self-attention head
        4. One layer of multi-head self-attention
        5. One block (which contains one layer of multi-head self-attention and FFN)
    
    Maybe the function name "sampling" is not good though... We may want to change this 
    name when the development is finished. 
    """
    
    # initializes a complete pre-trained model 
    # during the search, maybe set do_post_processing=False
    model = get_pretrained_model(model_name, device=device, num_classes=num_classes,
                                 with_classification_head=with_classification_head, 
                                 return_layer_head_count=False,
                                 ignore_logging=True, do_post_processing=False, 
                                 dataset_name=dataset_name)
    if granularity == "attention_head":
        # this is straightforward by using HuggingFace API
        
        # beware that after pruning, some model weights are no longer frozen,
        # remember to re-freeze them before fine-tuning. (TODO)
        
        # pruning_idx must be a dictionary mapping int to list of int
        model.prune_heads(heads_to_prune=pruning_idx)
    elif granularity == "block":
        # pruning_idx must be a list of int
        model.prune_blocks(blocks_to_prune=pruning_idx)
    elif granularity == "block_and_attention_head":
        # bi-granularity pruning (mixed block-wise and attention-head wise pruning)
        
        # pruning_idx must be a tuple of 2 elements:
        #   * the first is for pruning blocks (a list of int), 
        #   * the second is for pruning attention heads (a dictionary mapping int to list of int).
        blocks_to_prune, heads_to_prune = pruning_idx
        model.prune_heads(heads_to_prune=heads_to_prune)
        model.prune_blocks(blocks_to_prune=blocks_to_prune)
    elif granularity == "none":
        pass
    else:
        # other granularities may be considered later.
        raise Exception("Granularity {} not recognized.".format(granularity))
    
    return model


class DisableLoggerContextManager():
    # https://stackoverflow.com/a/20251235/7636942
    def __enter__(self):
        # https://docs.python.org/3/howto/logging.html
        # logging.WARNING is sufficient to suppress messages from get_pretrained_model
        logging.disable(logging.WARNING)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


class DummnyContextManager():
    def __enter__(self):
        pass

    def __exit__(self, exit_type, exit_value, exit_traceback):
        pass


