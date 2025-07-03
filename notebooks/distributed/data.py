import os
import torch
import torch.distributed as dist
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, DistributedSampler

def get_dataloaders(global_batch_size=256, download_dir="data/"):

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Path that torchvision uses to store processed data
    processed_train = os.path.join(download_dir, 'cifar-10-batches-py')

    if dist.get_rank() == 0:
        if not os.path.exists(processed_train):
            print("[Rank 0] Dataset not found, downloading CIFAR-10...")
            datasets.CIFAR10(download_dir, download=True, train=True, transform=transform)
            datasets.CIFAR10(download_dir, download=True, train=False, transform=transform)
        else:
            print("[Rank 0] Dataset already exists, skipping download.")

    ### TODO: Wait for other ranks to continue after download
    dist.barrier()

    world_size = dist.get_world_size()
    ### TODO: How many samples should each worker see?
    local_batch_size = global_batch_size // world_size

    trainset = datasets.CIFAR10(download_dir, train=True, transform=transform)
    testset = datasets.CIFAR10(download_dir, train=False, transform=transform)

    ### TODO: Ensure each rank gets unique shards of data
    train_sampler = DistributedSampler(trainset, shuffle=True)
    test_sampler = DistributedSampler(testset, shuffle=True)

    trainloader = DataLoader(trainset, batch_size=local_batch_size, sampler=train_sampler)
    testloader = DataLoader(testset, batch_size=local_batch_size, sampler=test_sampler)

    if dist.get_rank() == 0:
        print(f"[INFO] Dataset size: {len(trainset)} samples")
        print(f"[INFO] World size: {world_size} workers (data parallel)")
        print(f"[INFO] Global batch size: {global_batch_size}")
        print(f"[INFO] Local batch size per worker: {local_batch_size}")
        print(f"[INFO] Samples per worker per epoch: {len(trainset) // world_size}")

    ### TODO: Sync again among all ranks before training
    dist.barrier()

    return trainloader, testloader
