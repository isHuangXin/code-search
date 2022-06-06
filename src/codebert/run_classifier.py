""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""

import argparse
import logging
import torch
import os

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    # action='store_true'就代表着一旦有这个参数, 做出动作“将其值标为True”; 也就是没有时,默认状态下其值为False。
    # parser.add_argument("--data_dir", default=None, type=str, required=True,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    args = parser.parse_args()

    # set cuda, gpu & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        """ 
        Pytorch 分布式训练: 
                https://zhuanlan.zhihu.com/p/76638962
                https://zhuanlan.zhihu.com/p/166161217                
        """
        # torch.distributed.init_process_group(backend='nccl', world_size=torch.cuda.device_count(), rank=args.local_rank)
    args.device = device

    # setup logging
    logging.basicConfig(format="%(asctime)s - % (levelname) - %(name)s - %(message)s",
                        datefmt="%m%d%y %H:%M%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)


if __name__ == "__main__":
    main()
