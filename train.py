import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as f
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def main_worker(rank, world_size):

    # initialize the process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

    # create model and wrap DDP
    model = SimpleCNN().cuda(rank)
    model = DDP(model, device_ids=[rank])

    # prepare dataset + sample
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # training loop 
    for epoch in range(3):
        sampler.set_epoch(epoch)

        for image, labels in dataloader:
            images = images.cuda(rank)
            labels = labels.cuda(rank)

            optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    dist.destroy_process_group()

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        return self.fc2