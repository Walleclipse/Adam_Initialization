# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/main.py

from argparse import ArgumentParser
from warnings import simplefilter
import json
import os
import random
import sys
import tempfile
print('pid:', os.getpid())

from torch.multiprocessing import Process
import torch
import torch.multiprocessing as mp

import config
import loader
import utils.hdf5 as hdf5
import utils.log as log
import utils.misc as misc

RUN_NAME_FORMAT = ("{data_name}-" "{framework}-" "{phase}-" "{timestamp}")


def load_configs_initialize_training():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--entity", type=str, default=None, help="entity for wandb logging")
    parser.add_argument("--project", type=str, default='Optimization', help="project name for wandb logging")

    parser.add_argument("-cfg", "--cfg_file", type=str, default=None)
    parser.add_argument("-data", "--data_dir", type=str, default="data/")
    parser.add_argument("-save", "--save_dir", type=str, default="checkpoints/")
    parser.add_argument("-ckpt", "--ckpt_dir", type=str, default=None)
    parser.add_argument("-best", "--load_best", action="store_true", help="load the best performed checkpoint")

    parser.add_argument("--seed", type=int, default=0, help="seed for generating random numbers")
    parser.add_argument("-DDP", "--distributed_data_parallel", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl", help="cuda backend for DDP training \in ['nccl', 'gloo']")
    parser.add_argument("-tn", "--total_nodes", default=1, type=int, help="total number of nodes for training")
    parser.add_argument("-cn", "--current_node", default=0, type=int, help="rank of the current node")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("-sync_bn", "--synchronized_bn", action="store_true", help="turn on synchronized batchnorm")
    parser.add_argument("-mpc", "--mixed_precision", action="store_true", help="turn on mixed precision training")

    parser.add_argument("--truncation_factor", type=float, default=-1.0, help="truncation factor for applying truncation trick \
                        (-1.0 means not applying truncation trick)")
    parser.add_argument("--truncation_cutoff", type=float, default=None, help="truncation cutoff for stylegan \
                        (apply truncation for only w[:truncation_cutoff]")
    parser.add_argument("-batch_stat", "--batch_statistics", default=1, type=int, help="use the statistics of a batch when evaluating GAN \
                        (if false, use the moving average updated statistics)")
    parser.add_argument("-std_stat", "--standing_statistics", action="store_true", help="apply standing statistics for evaluation")
    parser.add_argument("-std_max", "--standing_max_batch", type=int, default=-1, help="maximum batch_size for calculating standing statistics \
                        (-1.0 menas not applying standing statistics trick for evaluation)")
    parser.add_argument("-std_step", "--standing_step", type=int, default=-1, help="# of steps for standing statistics \
                        (-1.0 menas not applying standing statistics trick for evaluation)")
    parser.add_argument("--freezeD", type=int, default=-1, help="# of freezed blocks in the discriminator for transfer learning")

    # parser arguments to apply langevin sampling for GAN evaluation
    # In the arguments regarding 'decay', -1 means not applying the decay trick by default
    parser.add_argument("-lgv", "--langevin_sampling", action="store_true",
                        help="apply langevin sampling to generate images from a Energy-Based Model")
    parser.add_argument("-lgv_rate", "--langevin_rate", type=float, default=-1,
                        help="an initial update rate for langevin sampling (\epsilon)")
    parser.add_argument("-lgv_std", "--langevin_noise_std", type=float, default=-1,
                        help="standard deviation of a gaussian noise used in langevin sampling (std of n_i)")
    parser.add_argument("-lgv_decay", "--langevin_decay", type=float, default=-1,
                        help="decay strength for langevin_rate and langevin_noise_std")
    parser.add_argument("-lgv_decay_steps", "--langevin_decay_steps", type=int, default=-1,
                        help="langevin_rate and langevin_noise_std decrease every 'langevin_decay_steps'")
    parser.add_argument("-lgv_steps", "--langevin_steps", type=int, default=-1, help="total steps of langevin sampling")

    parser.add_argument("-t", "--train", default=1, type=int)
    parser.add_argument("-hdf5", "--load_train_hdf5",  default=1, type=int, help="load train images from a hdf5 file for fast I/O")
    parser.add_argument("-l", "--load_data_in_memory", action="store_true", help="put the whole train dataset on the main memory for fast I/O")
    parser.add_argument("-metrics", "--eval_metrics", nargs='+', default=['fid'],
                        help="evaluation metrics to use during training, a subset list of ['fid', 'is', 'prdc'] or none")
    parser.add_argument("--resize_fn", type=str, default="legacy", help="which mode to use PIL.bicubic resizing for calculating clean metrics\
                        in ['legacy', 'clean']")
    parser.add_argument("-s", "--save_fake_images", action="store_true")
    parser.add_argument("-v", "--vis_fake_images", action="store_true", help=" visualize image canvas")
    parser.add_argument("-knn", "--k_nearest_neighbor", action="store_true", help="conduct k-nearest neighbor analysis")
    parser.add_argument("-itp", "--interpolation", action="store_true", help="conduct interpolation analysis")
    parser.add_argument("-fa", "--frequency_analysis", action="store_true", help="conduct frequency analysis")
    parser.add_argument("-tsne", "--tsne_analysis", action="store_true", help="conduct tsne analysis")
    parser.add_argument("-ifid", "--intra_class_fid", action="store_true", help="calculate intra-class fid")
    parser.add_argument('--GAN_train', action='store_true', help="whether to calculate CAS (Recall)")
    parser.add_argument('--GAN_test', action='store_true', help="whether to calculate CAS (Precision)")
    parser.add_argument('-resume_ct', '--resume_classifier_train', action='store_true', help="whether to resume classifier traning for CAS")
    parser.add_argument("-sefa", "--semantic_factorization", action="store_true", help="perform semantic (closed-form) factorization")
    parser.add_argument("-sefa_axis", "--num_semantic_axis", type=int, default=-1, help="number of semantic axis for sefa")
    parser.add_argument("-sefa_max", "--maximum_variations", type=float, default=-1,
                        help="iterpolate between z and z + maximum_variations*eigen-vector")

    parser.add_argument("--print_every", type=int, default=100, help="logging interval")
    parser.add_argument("-every", "--save_every", type=int, default=5000, help="save interval") # 2000
    parser.add_argument('--eval_backbone', type=str, default='Inception_V3', help="[SwAV, Inception_V3]")
    parser.add_argument("-ref", "--ref_dataset", type=str, default="test", help="reference dataset for evaluation[train/valid/test]")

    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer')
    parser.add_argument('--eps', type=float, default=1e-16, help='very small term epsilon in adaptive gradient methods')
    parser.add_argument('--device', default="cuda", type=str, )
    # init_optim_params
    parser.add_argument("--init_method_gen", default='none', type=str,
                        choices=['none','random','random-kaiming','grad-mean','grad-mean-var','grad-mean-random'])
    parser.add_argument("--init_method_disc", default='none', type=str,
                        choices=['none', 'random', 'random-kaiming', 'grad-mean', 'grad-mean-var', 'grad-mean-random'])
    parser.add_argument('--init_state', default="v", type=str,choices=['mv','m','v'])
    parser.add_argument('--init_scale', default=1.0, type=float,)
    parser.add_argument('--init_scale_m0', default=1.0, type=float,)
    parser.add_argument('--init_size', default=5000, type=int,)

    parser.add_argument('--save_name', default="result_new", type=str,)

    parser.add_argument('--optim_default_args', default=1, type=int,)
    args = parser.parse_args()

    if args.cfg_file is None:
        args.cfg_file = f"./configs/CIFAR10/DCGAN_{args.optimizer}.yaml"


    #torch.cuda.set_device(args.device)
    os.makedirs(args.save_name, exist_ok=True)
    #args.save_name = f"{save_name}/{args.model}_opt{args.alg}-{args.lr}-eps{args.eps}_init{args.init_method}-{args.init_state}-{args.init_scale}-{args.init_scale_m0}-{args.init_size}_seed{args.seed}"
    args.outf = f"{args.save_name}/gcgan_opt{args.optimizer}-{args.lr}-eps{args.eps}_init{args.init_method_gen}_{args.init_method_disc}-{args.init_state}-{args.init_scale}-{args.init_scale_m0}-{args.init_size}_seed{args.seed}"
    args.save_dir = f"{args.save_dir}/opt{args.optimizer}-{args.lr}-eps{args.eps}_init{args.init_method_gen}_{args.init_method_disc}-{args.init_state}-{args.init_scale}"
    print(args)

    #os.makedirs('result', exist_ok=True)
    #args.device = "cuda:0"

    run_cfgs = vars(args)

    if not args.train and \
            "none" in args.eval_metrics and \
            not args.save_fake_images and \
            not args.vis_fake_images and \
            not args.k_nearest_neighbor and \
            not args.interpolation and \
            not args.frequency_analysis and \
            not args.tsne_analysis and \
            not args.intra_class_fid and \
            not args.GAN_train and \
            not args.GAN_test and \
            not args.semantic_factorization:
        parser.print_help(sys.stderr)
        sys.exit(1)

    gpus_per_node, rank = torch.cuda.device_count(), torch.cuda.current_device()
    gpus_per_node, rank = 1, 0

    cfgs = config.Configurations(args.cfg_file)
    cfgs.update_cfgs(run_cfgs, super="RUN")
    cfgs.OPTIMIZATION.world_size = gpus_per_node * cfgs.RUN.total_nodes
    cfgs.OPTIMIZATION.eps = args.eps
    cfgs.check_compatability()

    run_name = log.make_run_name(RUN_NAME_FORMAT,
                                 data_name= cfgs.DATA.name,
                                 framework=cfgs.RUN.cfg_file.split("/")[-1][:-5],
                                 phase="train")

    crop_long_edge = False if cfgs.DATA.name in cfgs.MISC.no_proc_data else True
    resize_size = None if cfgs.DATA.name in cfgs.MISC.no_proc_data else cfgs.DATA.img_size
    if cfgs.RUN.load_train_hdf5:
        hdf5_path, crop_long_edge, resize_size = hdf5.make_hdf5(name=cfgs.DATA.name,
                                                                img_size=cfgs.DATA.img_size,
                                                                crop_long_edge=crop_long_edge,
                                                                resize_size=resize_size,
                                                                data_dir=cfgs.RUN.data_dir,
                                                                DATA=cfgs.DATA,
                                                                RUN=cfgs.RUN)
    else:
        hdf5_path = None
    cfgs.PRE.crop_long_edge, cfgs.PRE.resize_size = crop_long_edge, resize_size

    misc.prepare_folder(names=cfgs.MISC.base_folders, save_dir=cfgs.RUN.save_dir)
    misc.download_data_if_possible(data_name=cfgs.DATA.name, data_dir=cfgs.RUN.data_dir)

    if cfgs.RUN.seed == -1:
        cfgs.RUN.seed = random.randint(1, 4096)

    if cfgs.OPTIMIZATION.world_size == 1:
        print("You have chosen a specific GPU. This will completely disable data parallelism.")
    return cfgs, gpus_per_node, run_name, hdf5_path, rank


if __name__ == "__main__":
    cfgs, gpus_per_node, run_name, hdf5_path, rank = load_configs_initialize_training()

    if cfgs.RUN.distributed_data_parallel and cfgs.OPTIMIZATION.world_size > 1:
        mp.set_start_method("spawn", force=True)
        print("Train the models through DistributedDataParallel (DDP) mode.")
        try:
            torch.multiprocessing.spawn(fn=loader.load_worker,
                                        args=(cfgs,
                                              gpus_per_node,
                                              run_name,
                                              hdf5_path),
                                        nprocs=gpus_per_node)
        except KeyboardInterrupt:
            misc.cleanup()
    else:
        loader.load_worker(local_rank=rank,
                           cfgs=cfgs,
                           gpus_per_node=gpus_per_node,
                           run_name=run_name,
                           hdf5_path=hdf5_path)
