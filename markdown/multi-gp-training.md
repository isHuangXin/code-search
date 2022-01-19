# multi-gpu-training

PyTorch 为数据并行训练提供了几种选择:

- Step1: 如果数据和模型可以放在一个 GPU 中，并且不关心训练速度，使用单设备训练。
- Step2: 如果服务器上有多个 GPU，请使用单机多 GPU [DataParallel](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html), 并且您希望以最少的代码更改来加快训练速度。
- Step3: 如果您想进一步加快训练速度并愿意编写更多代码来设置它，请使用单机多 GPU [DistributedDataParallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)
- Step4: 如果应用需要跨计算机边界扩展，请使用多计算机[DistributedDataParallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html) 和 [启动脚本](https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md )。
- Step5: 如果预计会出现错误（例如，OOM），或者在训练过程中资源可以动态加入和离开，请使用[扭弹性](https://pytorch.org/elastic )启动分布式训练。

注意, 数据并行训练还可以与[自动混合精度（AMP）](https://pytorch.org/docs/master/notes/amp_examples.html#working-with-multiple-gpus )一起使用。

---

- torch.nn.DataParallel

DataParallel包以最低的编码障碍实现了单机多 GPU 并行处理。 它只需要一行更改应用代码。尽管DataParallel非常易于使用，但通常无法提供最佳性能。 这是因为DataParallel的实现会在每个正向传播中复制该模型，并且其单进程多线程并行性自然会遭受 GIL 争用。 为了获得更好的性能，请使用DistributedDataParallel。

- torch.nn.parallel.DistributedDataParallel

与DataParallel相比，DistributedDataParallel还需要设置一个步骤，即调用init_process_group相比。DDP 使用多进程并行性，因此在模型副本之间没有 GIL (全局解释器锁) 争用。 此外，该模型是在 DDP 构建时而不是在每个正向传播时广播的，这也有助于加快训练速度。 

