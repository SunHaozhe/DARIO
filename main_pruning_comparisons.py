"""
Adapted from main_train.py
Inspired by: 
https://github.com/VainF/Torch-Pruning/blob/master/examples/transformers/prune_hf_vit.py
https://github.com/VainF/Torch-Pruning/blob/master/benchmarks/main_imagenet.py
https://github.com/VainF/Torch-Pruning/blob/master/benchmarks/main.py
"""

import time
from data_loading import get_dataloaders
from nn_model import get_pretrained_preprocessor, get_pretrained_model
import torch.nn as nn
import torch_pruning as tp
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTSelfOutput
from transformers.models.mobilevit.modeling_mobilevit import MobileViTSelfAttention
from transformers.models.mobilevit.modeling_mobilevit import MobileViTSelfOutput
from transformers.models.mobilevit.modeling_mobilevit import MobileViTConvLayer
from utils import *


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()

    # general-purpose
    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true", default=False, 
                        help="whether to forbid using wandb to log the experiment.")
    parser.add_argument("--push_to_hub", action="store_true", default=False, 
                        help="whether to push fine-tuned model to Hub.")
    
    # training
    parser.add_argument("--lr", type=float, default=1e-3, 
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=300, 
                        help="how many epochs for fine-tuning.")
    parser.add_argument("--how_to_freeze", type=str, default=None, 
                        help="""How to freeze parameters before fine-tuning, see 
                        freeze_model_parameters() in nn_model.py""")
    
    # dataset
    parser.add_argument("--dataset", type=str, default="Icdar_Micro",
                        help="""["Icdar_Micro", "imagenette", "imagenet",  
                        "BCT_Micro", "BRD_Micro", "CRS_Micro", ...]""")
    parser.add_argument("--imagenet_root", type=str, default=None,
                        help="Path to the root folder of ImageNet.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="""how many subprocesses to use for data 
                        loading. 0 means that the data will be loaded 
                        in the main process.""")
    
    # model
    parser.add_argument("--model", type=str, default="MAE_ViT_base",
                        help="""["MAE_ViT_base", "MobileViT_small"]""")
    
    # pruning
    parser.add_argument("--pruning_type", 
                        default="lamp", type=str,
                        help="pruning algorithm", 
                        choices=["group_taylor", "lamp", "group_l2",
                                 "group_l2_sl", "l1", "greg_l1"])
    parser.add_argument("--pruning_ratio", 
                        type=float, default=0)
    parser.add_argument("--head_pruning_ratio", 
                        type=float, default=0)
    parser.add_argument("--prune_head_dims",
                        default=False, action="store_true")
    parser.add_argument("--prune_num_heads", 
                        default=False, action="store_true")
    parser.add_argument("--only_prune_transformer", 
                        default=False, action="store_true", 
                        help="If True, avoid pruning conv layers, only prune linear layers.")
    parser.add_argument('--global_pruning', 
                        default=False, action='store_true', help='global pruning')
    parser.add_argument('--prune_ViTSelfOutput', 
                        default=False, action='store_true', 
                        help="prune ViTSelfOutput or not, default is False (bottleneck).")
    parser.add_argument('--taylor_batchs', 
                        default=10, type=int, help='number of batchs for taylor criterion')
    parser.add_argument("--sparsity_learning_epochs", type=int, default=300)
    parser.add_argument("--sparsity_learning_lr", type=float, default=1e-3)
    parser.add_argument("--sparsity_learning_weight_decay",
                        type=float, default=0.01)
    parser.add_argument("--reg", 
                        type=float, default=1e-4, 
                        help="""regularization coefficient of the pruner""")
    parser.add_argument("--delta_reg", 
                        type=float, default=1e-4, 
                        help="""increment of regularization coefficient 
                        for growing regularization""")
    parser.add_argument("--precomputed_configuration", 
                        default=False, action="store_true")
    
    
    # continue training 
    parser.add_argument("--continue_training_checkpoint", type=str, default=None, 
                        help="""If None, continue training is not activated. 
                        Otherwise this argument should receive the checkpoint 
                        name. For example, exp_imagenet_200""")
    parser.add_argument("--continue_training_nb_jobs", type=int, default=None, 
                        help="""Used only if continue_training_checkpoint is not None. 
                        Specifies how many jobs would divide the total number 
                        of epochs, where the total number of epochs is set by 
                        num_epochs.""")
    parser.add_argument("--continue_training_job_rank", type=int, default=None,
                        help="""Used only if continue_training_checkpoint is not None. 
                        Specifies the rank/id of the job among consecutive jobs. 
                        Its value must be in [0, continue_training_nb_jobs - 1].""")
    
    args = parser.parse_args()
    args.use_wandb = not args.no_wandb
    
    if args.continue_training_checkpoint is not None:
        assert args.continue_training_nb_jobs is not None 
        assert args.continue_training_job_rank is not None
        
    if args.precomputed_configuration:
        if args.model == "MAE_ViT_base":
            args.pruning_ratio = 0.5
            args.prune_head_dims = True
            args.only_prune_transformer = True
        elif args.model == "MobileViT_small":
            args.pruning_ratio=0.86
            args.prune_head_dims=True
            args.only_prune_transformer=True
            
    return args


def init_wandb(args, verbose=True):
    if args.wandb_project_name is None:
        project_name = "DARIO_DepGraph_{}".format(args.dataset)
    else:
        project_name = args.wandb_project_name

    group_name = "{}_{}".format(args.model, args.pruning_type)
    
    if args.how_to_freeze is not None and args.how_to_freeze != "freeze_all_except_classification_head":
        group_name += "__{}".format(args.how_to_freeze)

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


def sparsity_train(train_dataloader, model, pruner, device, args):
    t0_sl = time.time()
    
    freeze_model_parameters(model, how="unfreeze_all")
    
    optim = torch.optim.AdamW(model.parameters(), lr=args.sparsity_learning_lr,
                              weight_decay=args.sparsity_learning_weight_decay)
    
    model.train()
    for epoch in range(1, args.sparsity_learning_epochs + 1):
        t0 = time.time()
        
        train_loss = 0
        acc = QuickAccuracyMetric()
        
        for data_item in train_dataloader:
            if isinstance(data_item, dict):
                # hf_image_collate_fn was used for the dataloader
                X, y = data_item["pixel_values"], data_item["labels"]
            else:
                # torch_dataloader_image_collate_fn was used for the dataloader
                X, y = data_item
            
            # transfer data to device (GPU)
            X, y = X.to(device), y.to(device)
            
            optim.zero_grad()

            logits = model(X).logits
            loss = F.cross_entropy(logits, y)
            loss.backward()
            
            # for sparsity learning
            pruner.regularize(model)  
            
            optim.step()
            
            train_loss += loss.item()
            acc.update(logits, y)

        train_loss /= len(train_dataloader)
        train_acc = acc.compute()
        
        t1 = time.time() - t0

        print("Sparsity learning: epoch {}, loss {:.2f}, accuracy {:.2f} %, epoch time {:.2f} s.".format(
            epoch, train_loss, train_acc, t1
        ))
        if args.use_wandb:
            wandb.log({
                "sl_epoch": epoch, 
                "sl_train_loss": train_loss,
                "sl_train_acc": train_acc,
                "sl_train_epoch_time": t1
            })
    
    model.eval()
    
    sl_total_time = time.time() - t0_sl
    print("Sparsity learning done in {:.2f} s.".format(sl_total_time))
    if args.use_wandb:
        wandb.run.summary["sl_total_time"] = sl_total_time
    
    return model


def prune_the_model(args, model):
    """
    Note that the pruned model unfreezes all parameters
    """
    if args.model == "MobileViT_small":
        example_inputs = torch.randn(1, 3, 256, 256).to(device)
    else:
        example_inputs = torch.randn(1, 3, 224, 224).to(device)
    
    nb_params_DG_unpruned, macs_DG_unpruned = measure_FLOPs_nb_params_DepGraph(
        model=model,
        example_inputs=example_inputs, 
        after_pruning=False, 
        use_wandb=args.use_wandb,
    )
    
    print("Pruning process starts....")
    
    t0_pruning = time.time()
    
    # DepGraph requires unfreezing all parameters
    freeze_model_parameters(model, how="unfreeze_all")
    
    need_sparsity_learning = False
    
    if args.pruning_type == "group_taylor":
        
        importance_criterion = tp.importance.GroupTaylorImportance(
            group_reduction="mean",
            normalizer="mean",
            multivariable=False,
            bias=False,
        )

        pruner_class = tp.pruner.MetaPruner
    elif args.pruning_type == "lamp":
        
        importance_criterion = tp.importance.LAMPImportance(
            p=2,
            group_reduction="mean",
            normalizer="lamp",
            bias=False,
        )

        pruner_class = tp.pruner.MagnitudePruner
    elif args.pruning_type == "group_l2":
        
        importance_criterion = tp.importance.GroupNormImportance(
            p=2,
            group_reduction="mean",
            normalizer="mean",
            bias=False,
        )
        
        pruner_class = tp.pruner.GroupNormPruner
    elif args.pruning_type == "group_l2_sl":
        
        need_sparsity_learning = True
        
        importance_criterion = tp.importance.GroupNormImportance(
            p=2,
            group_reduction="mean",
            normalizer="mean",
            bias=False,
        )
        
        pruner_class = partial(tp.pruner.GroupNormPruner, 
                               reg=args.reg)
    elif args.pruning_type == "l1":
        
        importance_criterion = tp.importance.MagnitudeImportance(
            p=1, 
            group_reduction="first",
            normalizer=None, 
            bias=False,
        )
        
        pruner_class = tp.pruner.MagnitudePruner
    elif args.pruning_type == "greg_l1":
        
        need_sparsity_learning = True
        
        importance_criterion = tp.importance.GroupNormImportance(
            p=1,
            group_reduction="mean",
            normalizer="mean",
            bias=False,
        )
        
        pruner_class = partial(tp.pruner.GrowingRegPruner, 
                               reg=args.reg,
                               delta_reg=args.delta_reg)
    else:
        raise NotImplementedError

    num_heads = {}
    ignored_layers = [model.classifier]
    # All heads should be pruned simultaneously, so we group channels by head.
    for m in model.modules():
        if isinstance(m, (ViTSelfAttention, MobileViTSelfAttention)):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads
        if (not args.prune_ViTSelfOutput) and isinstance(m, (ViTSelfOutput, MobileViTSelfOutput)):
            ignored_layers.append(m.dense)
            
        if args.only_prune_transformer and isinstance(m, (MobileViTConvLayer)):
            ignored_layers.append(m.convolution)
            if m.normalization is not None:
                ignored_layers.append(m.normalization)
    
    pruner = pruner_class(
        model,
        example_inputs,
        # If False, a uniform pruning ratio will be assigned to different layers.
        global_pruning=args.global_pruning,
        importance=importance_criterion,  # importance criterion for parameter selection
        pruning_ratio=args.pruning_ratio,  
        ignored_layers=ignored_layers,
        output_transform=lambda out: out.logits.sum(),
        num_heads=num_heads,
        prune_head_dims=args.prune_head_dims,
        prune_num_heads=args.prune_num_heads,
        head_pruning_ratio=args.head_pruning_ratio, # disabled when prune_num_heads=False
    )
    
    if need_sparsity_learning:
        model = sparsity_train(train_dataloader, model, pruner, device, args)
    
    # gradients need to be accumulated for each iteration if iterative pruning
    if isinstance(importance_criterion, tp.importance.TaylorImportance):
        t0_taylor_gradient_accumulation = time.time()
        
        freeze_model_parameters(model, how="unfreeze_all")
        
        model.zero_grad()
        print("Accumulating gradients for taylor pruning...")
        for k, data_item in enumerate(train_dataloader):
            if k >= args.taylor_batchs:
                break
            
            if isinstance(data_item, dict):
                # hf_image_collate_fn was used for the dataloader
                X, y = data_item["pixel_values"], data_item["labels"]
            else:
                # torch_dataloader_image_collate_fn was used for the dataloader
                X, y = data_item

            # transfer data to device (GPU)
            X, y = X.to(device), y.to(device)
            
            output = model(X).logits
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            
        taylor_gradient_total_time = time.time() - t0_taylor_gradient_accumulation
        
        print("Taylor gradient accumulation done in {:.2f} s.".format(
            taylor_gradient_total_time))
        if args.use_wandb:
            wandb.run.summary["taylor_gradient_total_time"] = taylor_gradient_total_time

    # the actual pruning happens here
    pruner.step()

    # Modify the attention head size and all head size aftering pruning
    for m in model.modules():
        if isinstance(m, (ViTSelfAttention, MobileViTSelfAttention)):
            print("num_heads:", m.num_attention_heads, 'head_dims:',
                m.attention_head_size, 'all_head_size:', m.all_head_size, '=>')
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
            print("num_heads:", m.num_attention_heads, 'head_dims:',
                m.attention_head_size, 'all_head_size:', m.all_head_size)
            print()
            
            if args.use_wandb:
                wandb.run.summary["num_attention_heads"] = m.num_attention_heads
                wandb.run.summary["attention_head_size"] = m.attention_head_size
                wandb.run.summary["all_head_size"] = m.all_head_size

    total_pruning_time = time.time() - t0_pruning
    
    print("Pruning process ends in {:.2f} s.".format(total_pruning_time))
    if args.use_wandb:
        wandb.run.summary["total_pruning_time"] = total_pruning_time
        
    nb_params_DG_pruned, macs_DG_pruned = measure_FLOPs_nb_params_DepGraph(
        model=model,
        example_inputs=example_inputs,
        after_pruning=True,
        use_wandb=args.use_wandb,
    )
    
    pruned_unpruned_ratio_nb_params = nb_params_DG_pruned / nb_params_DG_unpruned
    pruned_unpruned_ratio_FLOPs = macs_DG_pruned / macs_DG_unpruned
    print("Model size pruned/unpruned ratio = {:.2f} ; FLOPs pruned/unpruned ratio = {:.2f}".format(
        pruned_unpruned_ratio_nb_params,
        pruned_unpruned_ratio_FLOPs
    ))
    if args.use_wandb:
        wandb.run.summary["pruned_unpruned_ratio_nb_params"] = pruned_unpruned_ratio_nb_params
        wandb.run.summary["pruned_unpruned_ratio_FLOPs"] = pruned_unpruned_ratio_FLOPs
    
    return model


def measure_FLOPs_nb_params_DepGraph(model, example_inputs, after_pruning, use_wandb):
    macs_DG, nb_params_DG = tp.utils.count_ops_and_params(model, example_inputs)
    macs_DG = macs_DG / 1e9               # unit: G
    nb_params_DG = nb_params_DG / 1e6     # unit: M

    if after_pruning:
        print("Pruned (DepGraph measure). nb. params: {:.2f} M, MACs: {:.2f} G.".format(
            nb_params_DG, macs_DG
        ))
        if use_wandb:
            wandb.run.summary["nb_params_DG_pruned"] = nb_params_DG
            wandb.run.summary["macs_DG_pruned"] = macs_DG
    else:
        print("Unpruned (DepGraph measure). nb. params: {:.2f} M, MACs: {:.2f} G.".format(
            nb_params_DG, macs_DG
        ))
        if use_wandb:
            wandb.run.summary["nb_params_DG_unpruned"] = nb_params_DG
            wandb.run.summary["macs_DG_unpruned"] = macs_DG
            
    return nb_params_DG, macs_DG


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
        args, do_data_augmentation_for_training_split=True, image_processor=image_processor,
        collate_type="hf", device=device)
    
    if args.use_wandb:
        wandb.run.summary["nb_classes"] = num_classes
    
    if args.dataset in ["imagenet", "imagenette"]:
        eval_dataloader_ = val_dataloader
    else:
        eval_dataloader_ = test_dataloader
    
    model = get_pretrained_model(args.model, device=device, num_classes=num_classes,
                                 with_classification_head=True, return_layer_head_count=False,
                                 ignore_logging=False, do_post_processing=True,
                                 dataset_name=args.dataset)
    
    model = prune_the_model(args, model)
    
    fine_tuning(model=model, 
                train_dataset=train_dataloader.dataset, 
                eval_dataset=eval_dataloader_.dataset,
                per_device_train_batch_size=args.batch_size, 
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=args.num_epochs, 
                learning_rate=args.lr, 
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                use_wandb=args.use_wandb,
                how_to_freeze=args.how_to_freeze,
                run_name=wandb_run_name, 
                dataset_name=args.dataset, 
                continue_training_checkpoint=args.continue_training_checkpoint, 
                continue_training_nb_jobs=args.continue_training_nb_jobs,
                continue_training_job_rank=args.continue_training_job_rank)
    
    print("Test after fine-tuning:")
    test_model_classification_performance(
        test_dataloader=eval_dataloader_, model=model, device=device, use_wandb=args.use_wandb)
    
    if args.push_to_hub:
        fine_tuned_model_name = "pruned_{}_{}_DepGraph_{}".format(
            args.model, args.dataset, args.pruning_type)
        model.push_to_hub(
            repo_id="HzsAlhTransformerProject/{}".format(fine_tuned_model_name), private=True)
    
    overall_time = time.time() - t0_overall

    print("Done in {:.2f} s.".format(overall_time))
    if args.use_wandb:
        wandb.run.summary["overall_time"] = overall_time
