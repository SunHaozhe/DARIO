"""
Each run is identified by the combination of dataset, model, surrogate, 
and sampling random seed.

This script aims at collecting the following 4 metrics through wandb: 
* surrogate_value
* test_acc
* test_epoch_time
* test_inference_time_batch_avg

The Spearman correlation will be computed between surrogate_value and test_acc.
"""

import time
from collections import defaultdict
import torch.nn as nn
from data_loading import get_dataloaders
from nn_model import get_pretrained_preprocessor, get_pretrained_model, sampling
from zero_cost_nas.foresight.pruners.predictive import find_measures
from utils import *

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()

    # general-purpose
    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--sampling_random_seed", type=int, default=None, 
                        help="""random seed for the sampling of pruning index; 
                        no influence on the fine-tuning.""")
    parser.add_argument("--random_seed", type=int, default=None, 
                        help="""random seed for the fine-tuning; 
                        no influence on the sampling of pruning index.""")
    parser.add_argument("--no_wandb", action="store_true", default=False, 
                        help="whether to forbid using wandb to log the experiment.")
    parser.add_argument("--how_to_compute_surrogates", default="physical_pruning", 
                        help="""["physical_pruning", "nullifying_with_binary_mask"]""")
    parser.add_argument("--part", type=str, default="surrogate_computation",
                        choices=["fine_tuning", "surrogate_computation"], 
                        help="""(1) "surrogate_computation" computes surrogate values 
                        for several surrogates for a given sampling_random_seed. 
                        (2) "fine_tuning" gets classification performance for one 
                        given sampling_random_seed.""")
    
    # training
    parser.add_argument("--lr", type=float, default=1e-3, 
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=300, 
                        help="how many epochs for fine-tuning.")
    
    # dataset
    parser.add_argument("--dataset", type=str, default="Icdar_Micro",
                        help="""["Icdar_Micro", "imagenette", 
                        "BCT_Micro", "BRD_Micro", "CRS_Micro", ...]""")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="""how many subprocesses to use for data 
                        loading. 0 means that the data will be loaded 
                        in the main process.""")
    
    # model
    parser.add_argument("--model", type=str, default="MAE_ViT_base",
                        help="""["MAE_ViT_base", "MobileViT_small"]""")
    
    args = parser.parse_args()
    args.use_wandb = not args.no_wandb
    
    return args


def init_wandb(args, verbose=True):
    """
    "Correlation_{}_{}".format(args.dataset, args.model) was for 
    how_to_compute_surrogates=nullifying_with_binary_mask.
    
    "Correlation_v2_{}_{}".format(args.dataset, args.model) was for 
    how_to_compute_surrogates=physical_pruning.
    
    "Correlation_v3_{}_{}_{}".format(part, args.dataset, args.model) is for the 
    efficient implementation of Spearman correlation code. The semantic of 
    "synflow" changed (non-normalized by default).
    """
    if args.wandb_project_name is None:
        if args.part == "fine_tuning":
            part = "FT" 
        elif args.part == "surrogate_computation":
            part = "SC"
        project_name = "Correlation_v3_{}_{}_{}_nwft".format(part, args.dataset, args.model)
    else:
        project_name = args.wandb_project_name

    group_name = "{}".format(args.sampling_random_seed)

    wandb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb_logs")
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    wandb.init(entity="hzs_alh_transformer_project", 
               config=args, project=project_name,
               group=group_name, dir=wandb_dir)

    env_info = get_torch_gpu_environment()
    for k, v in env_info.items():
        wandb.run.summary[k] = v

    library_version = get_library_version()
    for k, v in library_version.items():
        wandb.run.summary[k] = v

    wandb_run_name = wandb.run.name

    if verbose:
        print("wandb_run_name: {}".format(wandb_run_name))
        print(env_info)
        print(library_version)
    return project_name, group_name, wandb_run_name


# the following function is from main_search.py
def get_performance_score(net, score, train_loader, num_classes, device, dataload="random", head_mask=None, alphas_blocks=None):
    return find_measures(net, train_loader, (dataload, 1, num_classes), device, head_mask=head_mask, alphas_blocks=alphas_blocks, measure_names=[score])[score]


def generate_random_head_mask(sampling_random_seed, num_layers, num_heads, device):
    rng = np.random.default_rng(sampling_random_seed)
    
    # discrete uniform distribution to generate a binary head mask
    # 1 indicates the head is not masked, 0 indicates the head is masked.
    head_mask = rng.integers(low=0, high=2, size=(
        num_layers, num_heads)).astype(np.float32)
    head_mask = torch.tensor(head_mask).to(device)
    return head_mask


def transform_mask_to_pruning_idx(head_mask, num_heads):
    heads_to_prune = defaultdict(list)
    for idx_block, head_mask_for_this_block in enumerate(head_mask):
        for idx_head, mask_for_this_head in enumerate(head_mask_for_this_block):
            mask_value = mask_for_this_head.item()
            if mask_value == 1:
                # we keep this head
                pass
            elif mask_value == 0:
                # we prune this head
                heads_to_prune[idx_block].append(idx_head)
            else:
                raise Exception("The head mask is not binary, please check.")

    # prune the block if all heads are pruned in that block
    blocks_to_prune = []
    new_heads_to_prune = dict()
    for idx_block, v in heads_to_prune.items():
        if len(v) == num_heads:
            # all heads are pruned in this block
            blocks_to_prune.append(idx_block)
        else:
            new_heads_to_prune[idx_block] = v
    if len(blocks_to_prune) > 0:
        granularity = "block_and_attention_head"
        pruning_idx = (blocks_to_prune, new_heads_to_prune)
    else:
        granularity = "attention_head"
        pruning_idx = new_heads_to_prune

    return pruning_idx, granularity


def get_surrogate_value(raw_surrogate_name, model_name, 
                        device, train_dataloader, num_classes, 
                        how_to_compute_surrogates, 
                        head_mask=None, pruning_idx=None, granularity=None):
    
    if raw_surrogate_name.startswith("normalized_"):
        surrogate_name = raw_surrogate_name[len("normalized_"):] 
    else:
        surrogate_name = raw_surrogate_name
    
    # load pruned model without classification head
    if how_to_compute_surrogates == "physical_pruning":
        assert pruning_idx is not None 
        assert granularity is not None 
        
        model = sampling(model_name=model_name,
                         granularity=granularity,
                         pruning_idx=pruning_idx,
                         device=device, num_classes=None,
                         with_classification_head=False)
        
        head_mask = None
    elif how_to_compute_surrogates == "nullifying_with_binary_mask":
        assert head_mask is not None 
        
        model = sampling(model_name=model_name,
                         granularity="none",
                         pruning_idx=None,
                         device=device, num_classes=None,
                         with_classification_head=False)
    else:
        raise Exception("how_to_compute_surrogates={} not recognized.".format(
            how_to_compute_surrogates))
    
    freeze_model_parameters(model, how="unfreeze_all")
    assert count_trainable_parameters(model) == count_parameters(model)
    
    if surrogate_name == "parameter_variance":
        surrogate_value = compute_model_parameter_variance(model)
    elif surrogate_name == "parameter_variance_v2":
        surrogate_value = compute_model_parameter_variance_v2(model)
    elif surrogate_name == "l0_norm_avg":
        surrogate_value = compute_Ln_norm_avg_surrogates(model, 0)
    elif surrogate_name == "l1_norm_avg":
        surrogate_value = compute_Ln_norm_avg_surrogates(model, 1)
    elif surrogate_name == "l2_norm_avg":
        surrogate_value = compute_Ln_norm_avg_surrogates(model, 2)
    elif surrogate_name == "l0_norm_avg_v2":
        surrogate_value = compute_Ln_norm_avg_v2_surrogates(model, 0)
    elif surrogate_name == "l1_norm_avg_v2":
        surrogate_value = compute_Ln_norm_avg_v2_surrogates(model, 1)
    elif surrogate_name == "l2_norm_avg_v2":
        surrogate_value = compute_Ln_norm_avg_v2_surrogates(model, 2)
    else:
        surrogate_value = get_performance_score(model, surrogate_name, train_dataloader, num_classes,
                                                device, head_mask=head_mask, alphas_blocks=None)
    
    if raw_surrogate_name.startswith("normalized_"):
        num_params = count_parameters(model)
        surrogate_value /= num_params
    
    surrogate_value = surrogate_value.item()
    
    return surrogate_value


def compute_model_parameter_variance(model):
    all_parameters = []
    for name, param in model.named_parameters():
        all_parameters.extend(torch.flatten(param.detach()).tolist())
    all_parameters = torch.tensor(all_parameters)
    res = torch.var(all_parameters)
    return res


def compute_model_parameter_variance_v2(model):
    """
    only considers weight (not bias) of linear 
    and conv2f layers. 
    This imitates the implementation of synflow
    """
    all_parameters = []
    for module in model.modules():
        # ignore layers like LayerNorm (like synflow)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # ignore bias, only consider weight (like synflow)
            param = module.weight
            all_parameters.extend(torch.flatten(param.detach()).tolist())
    all_parameters = torch.tensor(all_parameters)
    res = torch.var(all_parameters)
    return res


def compute_Ln_norm_avg_surrogates(model, n):
    """
    only considers weight (not bias) of linear 
    and conv2f layers. 
    This imitates the implementation of synflow
    """
    all_norms = []
    for module in model.modules():
        # ignore layers like LayerNorm (like synflow)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # ignore bias, only consider weight (like synflow)
            all_norms.append(module.weight.norm(n))

    all_norms = torch.tensor(all_norms)
    res = torch.mean(all_norms)
    return res


def compute_Ln_norm_avg_v2_surrogates(model, n):
    """
    Ln_norm_avg_v2 no longer imitates the implementation of synflow 
    because it seems to be suboptimal. 
    
    Ln_norm_avg_v2 imitates compute_model_parameter_variance
    """
    all_norms = []
    for name, param in model.named_parameters():
        all_norms.append(param.norm(n))
    
    all_norms = torch.tensor(all_norms)
    res = torch.mean(all_norms)
    return res


if __name__ == '__main__':    
    t0_overall = time.time()
    
    args = parse_arguments()
    
    if args.use_wandb:
        project_name, group_name, wandb_run_name = init_wandb(args)
    else:
        print(get_torch_gpu_environment())
        print(get_library_version())
        wandb_run_name = ""
    
    set_random_seeds(random_seed=args.random_seed)
    
    device = get_device(args)
    
    image_processor = get_pretrained_preprocessor(args)
    
    num_classes, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        args, do_data_augmentation_for_training_split=False, 
        image_processor=image_processor, collate_type="torch", device=device)
    
    #loading pretrained model to get properties and info
    _, num_layers, num_heads = get_pretrained_model(args.model, device=device, num_classes=num_classes,
                                                    with_classification_head=True,
                                                    return_layer_head_count=True,
                                                    ignore_logging=True, do_post_processing=False)
    
    head_mask = generate_random_head_mask(
        args.sampling_random_seed, num_layers, num_heads, device)
    
    pruning_idx, granularity = transform_mask_to_pruning_idx(
        head_mask=head_mask, num_heads=num_heads)
    
    if args.use_wandb:
        wandb.run.summary["head_mask"] = head_mask
        #wandb.run.summary["pruning_idx"] = pruning_idx #wandb error
        wandb.run.summary["granularity"] = granularity
    
    if args.part == "fine_tuning":
        num_classes, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            args, do_data_augmentation_for_training_split=True, 
            image_processor=image_processor, collate_type="torch", device=device)
        
        # load pruned model with classification head
        model = sampling(model_name=args.model,
                        granularity=granularity,
                        pruning_idx=pruning_idx,
                        device=device, num_classes=num_classes,
                        with_classification_head=True)
        
        fine_tuning(model=model, 
                    train_dataset=train_dataloader.dataset, 
                    eval_dataset=test_dataloader.dataset,
                    per_device_train_batch_size=args.batch_size, 
                    per_device_eval_batch_size=args.batch_size,
                    num_train_epochs=args.num_epochs, 
                    learning_rate=args.lr, 
                    weight_decay=args.weight_decay,
                    num_workers=args.num_workers,
                    use_wandb=args.use_wandb,
                    run_name=wandb_run_name)
        
        test_model_classification_performance(
            test_dataloader=test_dataloader, model=model, device=device, use_wandb=args.use_wandb)
        
    elif args.part == "surrogate_computation":
        
        # edit this list to get surrogate values for different surrogates
        raw_surrogate_name_list = ["l0_norm", "l1_norm", "l2_norm",
                                   "synflow", "normalized_synflow", "param_var",
                                   "composite_l0_var", "composite_l1_var", "composite_l0_l1"]
        
        for raw_surrogate_name in raw_surrogate_name_list:
            
            surrogate_value = get_surrogate_value(raw_surrogate_name=raw_surrogate_name,
                                                model_name=args.model,
                                                device=device,
                                                train_dataloader=train_dataloader,
                                                num_classes=num_classes,
                                                how_to_compute_surrogates=args.how_to_compute_surrogates,
                                                head_mask=head_mask,
                                                pruning_idx=pruning_idx,
                                                granularity=granularity)
            
            print("{} value = {}".format(raw_surrogate_name, surrogate_value))
            
            if args.use_wandb:
                wandb.run.summary["surrogate_{}".format(
                    raw_surrogate_name)] = surrogate_value
            
    overall_time = time.time() - t0_overall

    print("Done in {:.2f} s.".format(overall_time))
    if args.use_wandb:
        wandb.run.summary["overall_time"] = overall_time
