import torch
import torch.multiprocessing as mp
import os

from train import main_worker

def run():
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs.")

    mp.spawn(
        main_worker,
        nprocs=n_gpus,
        args=(n_gpus,),
        join=True
    )

if __name__ == "__main__":
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="12355"
    run()