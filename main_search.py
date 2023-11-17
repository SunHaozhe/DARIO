import argparse
import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from zero_cost_nas.foresight.pruners.predictive import find_measures
from data_loading import get_dataloaders
from nn_model import get_pretrained_preprocessor, get_pretrained_model, sampling
from tqdm import tqdm, trange

parser = argparse.ArgumentParser(description='Zero-Cost Pruning')
parser.add_argument("--dataset", type=str, default="ICDAR_micro",
                        help="""["imagenette", "BCT_Micro", "BRD_Micro", "CRS_Micro", ...]""")
parser.add_argument("--num_workers", type=int, default=10,
                        help="""how many subprocesses to use for data 
                        loading. 0 means that the data will be loaded 
                        in the main process.""")
parser.add_argument('--gpu', type=int, default=0, help="ID of the GPU to use for searching")
parser.add_argument('--model', type=str,
                    default="MAE_ViT_base", help="ViT model to prune")
parser.add_argument("--granularity", type=str, default="block", 
                        help="""["attention_head", "block", "block_and_attention_head"]""")
parser.add_argument('--nas_alg', type=str, default="diff", help="NAS algorithm to use [diff, rand]")
parser.add_argument('--score', type=str, default="param_var_v4", help="Score to choose for the neural network prediction: synflow/l1_norm/l0_norm")
parser.add_argument('--num_models', type=int, default=500, help="Number of models to sample")
parser.add_argument('--num_warm_start', type=int, default=100, help="Number of models to randomly sample for the warm start phase.")
parser.add_argument('--num_runs', type=int, default=1, help="Number of models to sample")
parser.add_argument('--threshold_blocks', type=float, default=0.3, help="Threshold for parsing blocks")
parser.add_argument('--threshold_heads', type=float, default=0.3, help="Threshold for parsing attention heads")
parser.add_argument('--img_size', type=int, default=224, help="Default image size used during the search process")
parser.add_argument('--lr', type=float, default=1e-1, help="Search optimizer learning rate")
parser.add_argument('--wd', type=float, default=1e-2, help="Search optimizer weight decay (L1 regularization)")
parser.add_argument('--batch_size', type=int, default=32, help="Minibatch size for the search process")
parser.add_argument('--seed', type=int, default=0)

#get the performance score score
def get_performance_score(net, score, train_loader, num_classes, device, dataload="random", head_mask=None, alphas_blocks=None):
    return find_measures(net, train_loader, (dataload, 1, num_classes), device, head_mask=head_mask, alphas_blocks=alphas_blocks, measure_names=[score])[score]


def idx_to_spec(sample, granularity, num_heads=None, threshold=0.6):
    """
    Returns the indices of heads/blocks to prune according to a given sigmoid vector sampled from the search space.
    """
    num_params = 0
    if granularity == "attention_head":
        d = {}
        for i, v in enumerate(sample):
            l = []
            best = (0,0)
            for j, p in enumerate(v):
                if p > best[1]:
                    best = (j,p)
                if p < threshold:
                    l.append(j)
                    if len(l) == num_heads: #If we decide to prune all heads from a layer, keep the best head to prevent unstability. Ideally we should prune the whole layer
                        l.remove(best[0])
                        num_params += 1
                else:
                    num_params += 1
            d[i] = l
        return d, num_params
    else:
        l = []
        for i,v in enumerate(sample):
            if v < threshold:
                l.append(i)
            else:
                num_params += 1
        return l, num_params

def plot_weights_heatmap(weights):
    fig, ax = plt.subplots()
    weights = weights.detach().cpu().numpy()
    im = ax.imshow(weights)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("weight value in [0,1]", rotation=-90, va="bottom")
   
    ax.legend()
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Attention head weight value when the search phase is over.")
    fig.tight_layout()
    plt.show()

def plot_blocks_barchart(weights):
    fig,ax = plt.subplots()
    weights = weights.detach().cpu().numpy()
    plt.bar(list(range(1, len(weights)+1)), weights, width=0.4)
    plt.ylabel("Sigmoid value")
    plt.title("Block weight value when the search phase is over.")
    plt.show()

def compute_score_stats(x):
    if len(x) == 1:
        return x[0], 0
    mean_ = np.mean(x, axis=0)
    low, high = scipy.stats.t.interval(0.95,
                                       len(x) - 1,
                                       loc=mean_,
                                       scale=scipy.stats.sem(x))

    confidence_interval = high - mean_

    return low, high, mean_, confidence_interval

def plot_score(scores, full_score, y_label, args, multi_runs=False):
    scores = np.array(scores)
    if multi_runs:
        s = scores.shape
        x = list(range(s[1]))
        lower_bound, upper_bound, scores, _ = compute_score_stats(scores)
        plt.plot(x, scores, 'k', color='#CC4F1B', label="pruned model")
        plt.fill_between(x, lower_bound, upper_bound, color='#CC4F1B', alpha=0.5)
    else:
        plt.plot(scores, 'k', color='#CC4F1B', label="pruned model")
    full_score = [full_score]*len(scores)
    plt.plot(full_score, label="original model")
    #plt.ylim([0.005653, 0.005659])
    plt.legend()
    plt.ylabel(y_label)
    plt.xlabel("Number of iterations")
    plt.tight_layout()
    plt.savefig(f"{args.nas_alg}_{args.granularity}_{args.num_runs}_runs_plot.pdf")

def random_search(train_loader, num_classes, num_layers, num_heads, device, args):
    scores_global = []

    net = sampling(args.model, "none", None, device)
    full_score = get_performance_score(net, args.score, train_loader, num_classes, device, head_mask=None, alphas_blocks=None).item()
    del net

    for j in range(args.num_runs):
        best_score = 1e10 if "param_var" in args.score else 0.0
        best_alphas = None
        torch.seed()
        if "cuda" in device:
            torch.cuda.seed()
        dist = torch.distributions.normal.Normal(0, 1)
        train_bar = trange(args.num_models)
        scores = []
        for i in train_bar:
            if args.granularity == "attention_head":
                sample = dist.rsample((num_layers, num_heads)).to(device=device)
                sample_heads = torch.sigmoid(sample)
                sample_blocks = None
            elif args.granularity == "block":
                sample = dist.rsample((num_layers,)).to(device=device)
                sample_blocks = torch.sigmoid(sample)
                sample_heads = None
            elif args.granularity == "block_and_attention_head":
                sample_heads = torch.sigmoid(dist.rsample((num_layers, num_heads)).to(device=device))
                sample_blocks = torch.sigmoid(dist.rsample((num_layers,)).to(device=device))
            else:
                raise Exception(f"Granularity {args.granularity} not supported.")
            net = sampling(args.model, "none", None, device)
            score = get_performance_score(net, args.score, train_loader, num_classes, device, head_mask=sample_heads, alphas_blocks=sample_blocks).item()
            scores.append(score)
            if "param_var" in args.score:
                if score < best_score:
                    best_score = score
                    best_alphas = (sample_blocks, sample_heads)
            elif score > best_score:
                best_score = score
                best_alphas = sample
            train_bar.set_description(f"Search iteration: [{i}/{args.num_models-1}], Score: {score}, Best Score: {best_score}")
        scores_global.append(scores)

    lower_bound, upper_bound, mean_, confidence_interval = compute_score_stats(scores_global)
    print(f"Average value: {np.mean(mean_)}")
    print(f"Confidence interval: {np.mean(confidence_interval)}")
    plot_score(scores_global, full_score, "Loss value (parameter variance)", args, multi_runs=(args.num_runs > 1))
    if sample_blocks is not None:
        plot_blocks_barchart(best_alphas[0])
    if sample_heads is not None:
        plot_weights_heatmap(best_alphas[1])
        if sample_blocks is not None:
            idx_dict_heads, _ = idx_to_spec(sample, "attention_head", num_heads, threshold=args.threshold_heads)
            idx_dict_blocks, _ = idx_to_spec(sample, "block", num_heads, threshold=args.threshold_blocks)
            best_idx_dict = (idx_dict_heads, idx_dict_blocks)
            for i in range(num_layers):
                if i in best_idx_dict[0]:
                    best_idx_dict[1].pop(i)
    save_path = f"genotypes/random_search_{args.model}_{args.granularity}.pt"
    torch.save({"idx_dict": best_idx_dict, "alphas_heads": best_alphas[1], "alphas_blocks": best_alphas[0]}, save_path)


def rand_warm_start(train_loader, num_classes, num_layers, num_heads, device, granularity, args):
    dist = torch.distributions.normal.Normal(0, 1)
    best_score = 1e10 if "param_var" in args.score else 0.0
    best_alphas = None
    train_bar = trange(args.num_warm_start)

    for i in train_bar:
        if granularity == "attention_head":
            sample = dist.rsample((num_layers, num_heads)).to(device=device)
            head_mask = torch.sigmoid(sample)
            alphas_blocks = None
            #threshold = args.threshold_heads
        else:
            sample = dist.rsample((num_layers,)).to(device=device)
            alphas_blocks = torch.sigmoid(sample)
            head_mask = None
            #threshold = args.threshold_blocks
        #idx_dict, _ = idx_to_spec(sample, granularity, num_heads, threshold=threshold)
        net = sampling(args.model, "none", None, device)
        score = get_performance_score(net, args.score, train_loader, num_classes, device, head_mask=head_mask, alphas_blocks=alphas_blocks).item()
        if "param_var" in args.score:
            if score < best_score:
                best_score = score
                best_alphas = sample
        elif score > best_score:
            best_score = score
            best_alphas = sample
        train_bar.set_description(f"Warm Starting ({granularity}): [{i}/{args.num_warm_start-1}], Score: {score}, Best Score: {best_score}")
    
    return best_alphas

def diff_search(train_loader, num_classes, num_layers, num_heads, device, args):
    scores_global = []

    net = sampling(args.model, "none", None, device)
    full_score = get_performance_score(net, args.score, train_loader, num_classes, device, head_mask=None, alphas_blocks=None).item()
    del net

    for j in range(args.num_runs):
        best_score = 1e10
        best_alphas = None
        best_idx_dict = None
        torch.seed()
        if "cuda" in device:
            torch.cuda.seed()
        param_list = []
        alphas_blocks = None
        alphas_heads = None
        if "attention_head" in args.granularity:
            alphas_heads = rand_warm_start(train_loader, num_classes, num_layers, num_heads, device, "attention_head", args)#torch.randn(num_layers, num_heads, requires_grad=True, device=device)
            alphas_heads.requires_grad = True
            param_list.append(alphas_heads)
        if "block" in args.granularity:
            alphas_blocks = rand_warm_start(train_loader, num_classes, num_layers, num_heads, device, "block", args) #torch.randn(num_layers, requires_grad=True, device=device)
            alphas_blocks.requires_grad = True
            param_list.append(alphas_blocks)
        maximize = False if "param_var" in args.score else True
        optmizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.wd, maximize=maximize)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optmizer, args.num_models)
        train_bar = trange(args.num_models)
        scores = []
        for i in train_bar:
            optmizer.zero_grad()
            if args.granularity == "attention_head":
                sample_heads = torch.sigmoid(alphas_heads)
                sample_blocks = None
                head_idx_dict, head_count = idx_to_spec(sample_heads, "attention_head", num_heads, threshold=args.threshold_heads)
                idx_dict = head_idx_dict
                count_str = f"Num Heads: {head_count}"
            elif args.granularity == "block":
                sample_heads = None
                sample_blocks = torch.sigmoid(alphas_blocks)
                block_idx_list, block_count = idx_to_spec(sample_blocks, "block", num_heads, threshold=args.threshold_blocks)
                idx_dict = block_idx_list
                count_str = f"Num Blocks: {block_count}"
            elif args.granularity == "block_and_attention_head":
                sample_heads = torch.sigmoid(alphas_heads)
                sample_blocks = torch.sigmoid(alphas_blocks)
                block_idx_list, block_count = idx_to_spec(sample_blocks, "block", num_heads, threshold=args.threshold_blocks)
                head_idx_dict, head_count = idx_to_spec(sample_heads, "attention_head", num_heads, threshold=args.threshold_heads)
                idx_dict = (block_idx_list, head_idx_dict)
                count_str = f"Num Heads: {head_count}, Num Blocks: {block_count}"
            else:
                raise Exception(f"Granularity {args.granularity} not supported.")
            net = sampling(args.model, "none", None, device)
            score = get_performance_score(net, args.score, train_loader, num_classes, device, head_mask=sample_heads, alphas_blocks=sample_blocks) #conversion to Python scalar is disabled in find_measures, the output are grad values so it should be differentiable
            score.backward()
            optmizer.step()
            scheduler.step()
            score = score.item()
            scores.append(score)
            train_bar.set_description(f"Search iteration: [{i}/{args.num_models-1}], Run: {j}/{args.num_runs-1}, Score: {score}, {count_str}")
            if score < best_score:
                best_score = score
                best_alphas = (sample_blocks, sample_heads)
                best_idx_dict = idx_dict
        scores_global.append(scores)
    
    lower_bound, upper_bound, mean_, confidence_interval = compute_score_stats(scores_global)
    print(f"Average value: {np.mean(mean_)}")
    print(f"Confidence interval: {np.mean(confidence_interval)}")
    plot_score(scores_global, full_score, "Loss value (parameter variance)", args, multi_runs=(args.num_runs > 1))
    if alphas_blocks is not None:
        plot_blocks_barchart(best_alphas[0])
    if alphas_heads is not None:
        plot_weights_heatmap(best_alphas[1])
        print(f"Num heads: {head_count}")
        if alphas_blocks is not None:
            for i in range(num_layers):
                if i in best_idx_dict[0]:
                    best_idx_dict[1].pop(i)
    save_path = f"genotypes/diff_search_{args.model}_{args.granularity}_{args.threshold_blocks}.pt"
    torch.save({"idx_dict": best_idx_dict, "alphas_heads": best_alphas[1], "alphas_blocks": best_alphas[0]}, save_path)

if __name__ == '__main__':
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
        if args.seed != 0:
            torch.cuda.manual_seed(args.seed)
    else:
        device = "cpu"
    if args.seed != 0:
        torch.manual_seed(args.seed)
        
    #loading dataset
    image_processor = get_pretrained_preprocessor(args)
    num_classes, train_loader, _, _ = get_dataloaders(
        args, do_data_augmentation_for_training_split=False, image_processor=image_processor, collate_type="torch", device=device)
        
    #loading pretrained model to get properties and info 
    _, num_layers, num_heads = get_pretrained_model(args.model, device=device, num_classes=num_classes, 
                                                        with_classification_head=True, 
                                                        return_layer_head_count=True,
                                                        ignore_logging=False, do_post_processing=True) 
   
    if args.nas_alg == "diff":
        diff_search(train_loader, num_classes, num_layers, num_heads, device, args)
    elif args.nas_alg == "rand":
        random_search(train_loader, num_classes, num_layers, num_heads, device, args)
    else:
        raise ValueError("Unsupported NAS algorithm.")

 
