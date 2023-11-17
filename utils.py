import os
import time
import random
import math
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from transformers import TrainingArguments, Trainer
import transformers
import datasets

from data_loading import hf_image_collate_fn
from nn_model import freeze_model_parameters

from prunable_model_classes import MobileViTForImageClassificationPrunableCutmix

def count_trainable_parameters(model):
    return sum([x.numel() for x in model.parameters() if x.requires_grad])

def count_parameters(model):
    return sum([x.numel() for x in model.parameters()])

def get_device(args=None, verbose=True):
    force_cpu = "False"
    if args is not None and hasattr(args, "force_cpu"):
        force_cpu = args.force_cpu

    if torch.cuda.is_available() and force_cpu == "False":
        device = torch.device("cuda")
        if verbose:
            print("Using GPU for PyTorch: {}".format(
                torch.cuda.get_device_name(
                    torch.cuda.current_device())))
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU for PyTorch")
    return device


def set_random_seeds(random_seed):
    if random_seed is not None:
        torch.backends.cudnn.deterministic = True
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)


def get_torch_gpu_environment():
    env_info = dict()
    env_info["PyTorch_version"] = torch.__version__

    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["cuDNN_version"] = torch.backends.cudnn.version()
        env_info["nb_available_GPUs"] = torch.cuda.device_count()
        env_info["current_GPU_name"] = torch.cuda.get_device_name(
            torch.cuda.current_device())
    else:
        env_info["nb_available_GPUs"] = 0
    return env_info


def get_library_version():
    import sys

    library_check_list = ["torch", "transformers", "datasets"]

    library_version = dict()
    for library_name in library_check_list:
        if library_name in sys.modules:
            library_version["{}_version".format(
                library_name)] = globals()[library_name].__version__
    return library_version


def report_trainable_params_count(model, use_wandb):
    trainable_params = count_trainable_parameters(model)
    total_params = count_parameters(model)
    print("count_trainable_parameters(model) = {}".format(trainable_params))
    print("count_parameters(model) = {}".format(total_params))
    if use_wandb:
        wandb.run.summary["count_trainable_parameters"] = trainable_params
        wandb.run.summary["count_parameters"] = total_params


def _hf_compute_metrics(eval_pred, metric):
    """
    eval_pred: 
        transformers.trainer_utils.EvalPrediction
        https://huggingface.co/docs/transformers/internal/trainer_utils#transformers.EvalPrediction
    logits: 
        numpy.ndarray, shape=(nb_samples, nb_classes), float32
    labels: 
        numpy.ndarray, shape=(nb_samples,), int64
    predictions: 
        numpy.ndarray, shape=(nb_samples,), int64
    result: 
        python dict with the key "accuracy"
        e.g. {'accuracy': 0.0925}
    nb_samples is the total number of examples in a dataset (e.g. test split)
    
    metric:
        QuickAccuracyMetric
    
    There are some issues of using HuggingFace's evaluate module with distributed settings or clusters:
        https://github.com/huggingface/datasets/issues/1942 
        https://huggingface.co/docs/evaluate/a_quick_tour#distributed-evaluation 
        
        Maybe it can be fixed by doing metric=evaluate.load("accuracy", keep_in_memory=True), but not 
        sure. Luckily, the accuracy computation is simple, we do not need to use HF API for small datasets.
    """
    logits, labels = eval_pred
    metric.update(torch.from_numpy(logits), torch.from_numpy(labels))
    
    accuracy_ = metric.compute()
    result = {"accuracy": accuracy_}
    return result


def fine_tuning(model, train_dataset, eval_dataset, 
                per_device_train_batch_size, 
                per_device_eval_batch_size,
                num_train_epochs, learning_rate, weight_decay, 
                num_workers, use_wandb, how_to_freeze=None, 
                run_name=None, dataset_name=None, 
                continue_training_checkpoint=None, 
                continue_training_nb_jobs=None,
                continue_training_job_rank=None):
    if how_to_freeze is None:
        # freeze parameters before fine-tuning (some parameters become unfrozen after pruning)
        freeze_model_parameters(
            model, how="freeze_all_except_classification_head")
    else:
        freeze_model_parameters(model, how=how_to_freeze)
        
    report_trainable_params_count(model, use_wandb=use_wandb)
    
    # set up the evaluation metric (we intentionally avoid using HuggingFace's evaluate library)
    metric = QuickAccuracyMetric()
    partial_compute_metrics = partial(_hf_compute_metrics, metric=metric)
    
    # set up hyperparameters and training options
    
    trainer_output_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "trainer_output_dir")
    
    if continue_training_checkpoint is not None:
        trainer_output_dir = os.path.join(
            trainer_output_dir, continue_training_checkpoint)
    
    if use_wandb:
        trainer_report_to = "wandb"
    else:
        trainer_report_to = "none"
        
    if dataset_name is not None and dataset_name == "imagenet":
        lr_scheduler_type_ = "cosine"
        evaluation_strategy_ = "no"
        fp16_ = True
    else:
        lr_scheduler_type_ = "constant"
        evaluation_strategy_ = "epoch"
        fp16_ = False
        
    if continue_training_checkpoint is not None:
        save_strategy_ = "epoch"
        
        nb_epochs_per_job = int(math.ceil(num_train_epochs / continue_training_nb_jobs))
        continue_tr_num_train_epochs = nb_epochs_per_job * (continue_training_job_rank + 1)
        continue_tr_num_train_epochs = min(continue_tr_num_train_epochs, num_train_epochs)
        
        num_train_epochs = continue_tr_num_train_epochs
    else:
        save_strategy_ = "no"
    
    training_args = TrainingArguments(
        # hyperparameters
        per_device_train_batch_size=per_device_train_batch_size,
        optim="adamw_torch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type_,
        num_train_epochs=num_train_epochs,
        # evaluation
        evaluation_strategy=evaluation_strategy_,
        per_device_eval_batch_size=per_device_eval_batch_size,
        # logging
        logging_strategy="epoch",
        logging_first_step=True,
        logging_nan_inf_filter=False,
        # save
        save_strategy=save_strategy_,
        # misc
        output_dir=trainer_output_dir,
        run_name=run_name,
        dataloader_num_workers=num_workers,
        fp16=fp16_,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=trainer_report_to,
        load_best_model_at_end=False,
        torch_compile=False,
        
        # full_determinism=True may cause some errors, 
        # furthermore torch.use_deterministic_algorithms is in beta
        full_determinism=False,  
    )
    
    # maybe a more elegant way to do the following to avoid code duplication
    if continue_training_checkpoint is not None and continue_training_job_rank > 0:
        training_args = TrainingArguments(
            # new arguments for continual training,
            # the other arguments are just copy-paste from the above
            overwrite_output_dir=True,
            resume_from_checkpoint=True,
            
            # hyperparameters
            per_device_train_batch_size=per_device_train_batch_size,
            optim="adamw_torch",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type_,
            num_train_epochs=num_train_epochs,
            # evaluation
            evaluation_strategy=evaluation_strategy_,
            per_device_eval_batch_size=per_device_eval_batch_size,
            # logging
            logging_strategy="epoch",
            logging_first_step=True,
            logging_nan_inf_filter=False,
            # save
            save_strategy=save_strategy_,
            # misc
            output_dir=trainer_output_dir,
            run_name=run_name,
            dataloader_num_workers=num_workers,
            fp16=fp16_,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=trainer_report_to,
            load_best_model_at_end=False,
            torch_compile=False,

            # full_determinism=True may cause some errors,
            # furthermore torch.use_deterministic_algorithms is in beta
            full_determinism=False,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=hf_image_collate_fn,
        compute_metrics=partial_compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=None,
    )
    
    # fine-tuning begins
    if continue_training_checkpoint is not None and continue_training_job_rank > 0:
        train_results = trainer.train(
            resume_from_checkpoint=True)
    else:
        train_results = trainer.train()
    
    return trainer, train_results


def test_model_classification_performance(test_dataloader, model, device, use_wandb, 
                                          acc_logging_name="test_acc"):
    """
    This simple test function avoids the complexity of HuggingFace framework.
    This function measures accuracy as well as inference speed.
    """

    t0 = time.time()

    t_inference_batch_sum = 0

    acc = QuickAccuracyMetric()

    model.eval()
    with torch.no_grad():
        for data_item in test_dataloader:
            
            if isinstance(data_item, dict):
                # hf_image_collate_fn was used for the dataloader
                X, y = data_item["pixel_values"], data_item["labels"]
            else:
                # torch_dataloader_image_collate_fn was used for the dataloader
                X, y = data_item
            
            # transfer data to device (GPU)
            X, y = X.to(device), y.to(device)

            t0_inference = time.time()

            if isinstance(model, MobileViTForImageClassificationPrunableCutmix):
                # for ImageNet experiments (cutmix data augmentation)
                logits = model.forward_test(X).logits
            else:
                logits = model(X).logits
            
            proba = F.softmax(logits, dim=1)

            t_inference_batch = time.time() - t0_inference

            t_inference_batch_sum += t_inference_batch

            acc.update(proba, y)

    test_acc = acc.compute()

    t1 = time.time() - t0

    t_inference_batch_avg = t_inference_batch_sum / len(test_dataloader)

    print("Test acc {:.2f}% | Time {:.1f} seconds. Avg batch inference time {:.3f} seconds.".format(
        test_acc, t1, t_inference_batch_avg))

    if use_wandb:
        wandb.run.summary[acc_logging_name] = test_acc
        wandb.run.summary["test_epoch_time"] = t1
        wandb.run.summary["test_inference_time_batch_avg"] = t_inference_batch_avg


class QuickAccuracyMetric:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.correct_cnt = 0
        self.total_cnt = 0

    def update(self, logits, target):
        self.correct_cnt += (logits.argmax(dim=1) == target).sum().item()
        self.total_cnt += target.shape[0]

    def compute(self, format="percentage"):
        res = self.correct_cnt / self.total_cnt
        if format == "percentage":
            res = res * 100
        return res


def filter_empty_lists_in_dict(pruning_idx):
    new_pruning_idx = dict()
    for k, v in pruning_idx.items():
        if len(v) > 0:
            new_pruning_idx[k] = v
    return new_pruning_idx


def threshold_block_granularity(loaded_object, threshold):
    block_alphas = loaded_object["alphas_blocks"].tolist()

    res = []
    for idx, alpha_ in enumerate(block_alphas):
        if alpha_ < threshold:
            res.append(idx)
    return res


def count_pruned_blocks(pruning_idx):
    assert isinstance(pruning_idx, list)
    return len(pruning_idx)

