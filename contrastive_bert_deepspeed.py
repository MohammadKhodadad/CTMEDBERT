import torch
import tqdm
import deepspeed
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.loss import InfoNCELoss
from utils.contrastive_data_loader import get_contrastive_dataloader
from utils.dataloader.mimic import create_mimic_cl_data_from_csv
from utils.dataloader.pubmed import download_pubmed_cl
from utils.dataloader.trialsgov import create_trials_contrastive_learning_data
from utils.optimizer import get_optimizer_and_scheduler

# DeepSpeed configuration
DS_CONFIG = {
    "train_batch_size": 64,  # Adjust batch size as per GPU memory
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,  # ZeRO stage 2 optimization
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
    },
    "gradient_accumulation_steps": 8,  # Accumulate gradients to simulate larger batch sizes
}

EPOCHS = 100
SAVE_STEP = 5000
WARM_UP_STEPS = 1000
TOTAL_STEPS = 100000
LEARNING_RATE = 0.00005

# Load tokenizer and model
tokenizer = load_tokenizer("bert-base-uncased")
model = Model("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/mlm/step_130000")

# Load data
train_loader, test_loader = get_contrastive_dataloader('./data/csvs', tokenizer)

# Loss function
criterion = InfoNCELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get optimizer and scheduler from utility function
optimizer, scheduler = get_optimizer_and_scheduler(model, LEARNING_RATE, WARM_UP_STEPS, TOTAL_STEPS)

# DeepSpeed initialization
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=DS_CONFIG,
    optimizer=optimizer
)

step = 0
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1} (Training)")
    
    for batch1, batch2 in progress_bar:
        batch1 = {key: batch1[key].to(device) for key in batch1.keys()}
        batch2 = {key: batch2[key].to(device) for key in batch2.keys()}
        
        outputs1 = model.encode(batch1)
        outputs2 = model.encode(batch2)
        loss = criterion(outputs1, outputs2)
        
        model.backward(loss)  # DeepSpeed backward pass
        model.step()  # DeepSpeed optimizer step
        scheduler.step()  # Manual scheduler step
        
        total_loss += loss.item()
        step += 1
        
        if step % SAVE_STEP == 0:
            print(f'Saving model at step {step}')
            model.save_checkpoint(f'./weights/contrastive/step_{step}/')
        
        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix({'Step': step, "Loss": avg_loss})
    
    print(f"Epoch {epoch + 1}, Step: {step}, Loss: {avg_loss}")
    
    model.eval()
    total_eval_loss = 0
    progress_bar = tqdm.tqdm(test_loader, desc=f"Epoch {epoch + 1} (Testing)")
    
    with torch.no_grad():
        for batch1, batch2 in progress_bar:
            batch1 = {key: batch1[key].to(device) for key in batch1.keys()}
            batch2 = {key: batch2[key].to(device) for key in batch2.keys()}
            
            outputs1 = model.encode(batch1)
            outputs2 = model.encode(batch2)
            
            loss = criterion(outputs1, outputs2)
            total_eval_loss += loss.item()
            
            avg_eval_loss = total_eval_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({"Eval Loss": avg_eval_loss})
    
    print(f"Epoch {epoch + 1}, Evaluation Loss: {avg_eval_loss}")