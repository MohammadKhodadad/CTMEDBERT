import tqdm
import torch
import pandas as pd

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.loss import InfoNCELoss
from utils.contrastive_data_loader import get_contrastive_dataloader
from utils.optimizer import get_optimizer_and_scheduler
from utils.dataloader.mimic import create_mimic_cl_data_from_csv
from utils.dataloader.pubmed import download_pubmed_cl
from utils.dataloader.trialsgov import create_trials_contrastive_learning_data

EPOCHS=100
WARM_UP_STEPS= 1000
TOTAL_STEPS = 100000
SAVE_STEP= 5000
BATCH_SIZE = 64

import os # Apparently these 3 are important to handle semaphore leakage
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    tokenizer = load_tokenizer("bert-base-uncased")
    # model = Model("bert-base-uncased")
    model = Model("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/mlm/step_60000")
    model = DDP(model, device_ids=[rank])
    # print('Handling Mimic Data')
    # create_mimic_cl_data_from_csv('./data/discharge_processed.csv','./data/csvs','discharge_diagnosis',['chief_complaint','history_of_present_illness'])
    # print('Handling PubMed Data')
    # download_pubmed_cl('./data/csvs')
    # print('Handling Trials Data')
    # create_trials_contrastive_learning_data('./data/clinical_trials_all_studies.csv','./data/csvs')

    train_loader, test_loader = get_contrastive_dataloader('./data/csvs', tokenizer, batch_size=BATCH_SIZE, distributed=True, rank=rank, world_size=world_size)
    criterion = InfoNCELoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model,0.00005,WARM_UP_STEPS, TOTAL_STEPS)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    step=0
    for epoch in range(EPOCHS):
        total_loss=0
        model.train()
        train_sampler.set_epoch(epoch)
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}: Learning Rate = {param_group['lr']}")
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1} (Training)", disable=(rank != 0))
        for batch1, batch2 in progress_bar:
            batch1 = {key: batch1[key].to(rank) for key in batch1.keys()}
            batch2 = {key: batch2[key].to(rank) for key in batch2.keys()}
            outputs1 = model.encode(batch1)
            outputs2 = model.encode(batch2)
            loss = criterion(outputs1, outputs2)
            total_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step+=1
            
            avg_loss = total_loss / (progress_bar.n + 1)
            if rank == 0:
                progress_bar.set_postfix({'Step':step,"Loss": avg_loss})
                if step>0 and step%SAVE_STEP==0:
                print(f'Saving model in step: {step}')
                model.save_pretrained(f'./weights/contrastive/step_{step}/')
        if rank==0:
            print(f"Epoch {epoch + 1}, Step: {step}, Loss: {avg_loss}")
        model.eval()
        total_eval_loss = 0
        progress_bar = tqdm.tqdm(test_loader, desc=f"Epoch {epoch + 1} (Testing)", disable=(rank != 0))
    
        with torch.no_grad():
            for batch1, batch2 in progress_bar:
                # Move batches to the device
                batch1 = {key: batch1[key].to(rank) for key in batch1.keys()}
                batch2 = {key: batch2[key].to(rank) for key in batch2.keys()}
                
                # Forward pass: encode both batches
                outputs1 = model.encode(batch1)
                outputs2 = model.encode(batch2)
                
                # Compute contrastive loss
                loss = criterion(outputs1, outputs2)
                total_eval_loss += loss.item()
                
                avg_eval_loss = total_eval_loss / (progress_bar.n + 1)
                if rank == 0:
                    progress_bar.set_postfix({"Eval Loss": avg_eval_loss})
            if rank == 0:
                print(f"Epoch {epoch + 1}, Evaluation Loss: {avg_eval_loss}")

    cleanup()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
