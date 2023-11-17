import time
from data_loading import get_dataloaders
from nn_model import get_pretrained_preprocessor, get_pretrained_model, sampling
from utils import *


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()

    # general-purpose
    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true", default=False, 
                        help="whether to forbid using wandb to log the experiment.")
    parser.add_argument("--no_push_to_hub", action="store_true", default=False, 
                        help="whether to forbid pushing fine-tuned model to Hub.")
    
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
    parser.add_argument("--granularity", type=str, default="block", 
                        help="""["attention_head", "block", "block_and_attention_head"]""")
    
    # load pruned models
    parser.add_argument("--pruning_idx_path", type=str, default=None, 
                        help="""If None (default), load the full model. Otherwise, 
                        this needs to be a path to a .pt file, which contains 
                        pruning_idx.""")
    parser.add_argument("--threshold", type=float, default=None, 
                        help="""This argument is optional. The default (None) 
                        will use the default threshold embedded in pruning_idx_path. 
                        Setting this argument to a float will override the threshold 
                        embedded in pruning_idx_path.""")
    
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
    args.push_to_hub = not args.no_push_to_hub
    
    if args.pruning_idx_path is not None:
        # do upload fine-tuned pruned models
        args.push_to_hub = False
    
    if args.continue_training_checkpoint is not None:
        assert args.continue_training_nb_jobs is not None 
        assert args.continue_training_job_rank is not None
    
    return args


def init_wandb(args, verbose=True):
    if args.wandb_project_name is None:
        if args.threshold is None:
            project_name = "FineTuning_{}_nwft".format(args.dataset)
        else:
            project_name = "FineTuning_thresholds_{}_nwft".format(args.dataset)
    else:
        project_name = args.wandb_project_name

    if args.pruning_idx_path is None:
        group_name = "{}".format(args.model)
    else:
        pruning_idx_basename = os.path.basename(args.pruning_idx_path)
        group_name = "{}_{}".format(args.model, pruning_idx_basename)
        
    if args.threshold is not None:
        group_name += "___{}".format(args.threshold)
        
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
    
    if args.pruning_idx_path is None:
        model = get_pretrained_model(args.model, device=device, num_classes=num_classes, 
                                    with_classification_head=True, return_layer_head_count=False, 
                                    ignore_logging=False, do_post_processing=True, 
                                    dataset_name=args.dataset)
    else:
        pruning_idx_basename = os.path.basename(args.pruning_idx_path)
        
        pruning_idx_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            args.pruning_idx_path
        )
        loaded_object = torch.load(pruning_idx_path, map_location=device)
        pruning_idx = loaded_object["idx_dict"]
        
        if "block" in pruning_idx_basename and "head" not in pruning_idx_basename:
            granularity_ = "block"
        elif "block" not in pruning_idx_basename and "head" in pruning_idx_basename:
            granularity_ = "attention_head"
            
            pruning_idx = filter_empty_lists_in_dict(pruning_idx)
        elif "block" in pruning_idx_basename and "head" in pruning_idx_basename:
            granularity_ = "block_and_attention_head"
            
            blocks_to_prune, heads_to_prune = pruning_idx
            heads_to_prune = filter_empty_lists_in_dict(heads_to_prune)
            pruning_idx = (blocks_to_prune, heads_to_prune)
        else:
            raise Exception("Cannot infer granularity from {}".format(
                args.pruning_idx_path))
            
        if args.threshold is not None:
            # override pruning_idx
            assert granularity_ == "block", "this only supports block granularity now."
            pruning_idx = threshold_block_granularity(
                loaded_object, threshold=args.threshold)
            
            if args.use_wandb:
                wandb.run.summary["count_pruned_blocks"] = count_pruned_blocks(pruning_idx)
            
        # load pruned model with classification head
        model = sampling(model_name=args.model,
                         granularity=granularity_,
                         pruning_idx=pruning_idx,
                         device=device, num_classes=num_classes,
                         with_classification_head=True, 
                         dataset_name=args.dataset)
        
    if args.use_wandb:
        wandb.run.summary["nb_classes"] = num_classes
        
    if args.dataset in ["imagenet", "imagenette"]:
        eval_dataloader_ = val_dataloader
    else:
        eval_dataloader_ = test_dataloader
    
    print("Test before fine-tuning:")
    test_model_classification_performance(
        test_dataloader=eval_dataloader_, model=model, device=device, use_wandb=args.use_wandb,
        acc_logging_name="test_acc_before_FineTuning")
    
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
        fine_tuned_model_name = "{}_{}_FineTuning".format(args.model, args.dataset)
        model.push_to_hub(
            repo_id="HzsAlhTransformerProject/{}".format(fine_tuned_model_name), private=True)
    
    overall_time = time.time() - t0_overall

    print("Done in {:.2f} s.".format(overall_time))
    if args.use_wandb:
        wandb.run.summary["overall_time"] = overall_time
