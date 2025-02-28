import torch
import tqdm
import deepspeed
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.loss import SupervisedInfoNCELoss
from utils.supervised_contrastive_data_loader import get_supervised_contrastive_dataloader
from utils.optimizer import get_optimizer_and_scheduler



EPOCHS = 100
SAVE_STEP = 8000
WARM_UP_STEPS = 1000
TOTAL_STEPS = 100000
LEARNING_RATE = 0.00005
BATCH_SIZE = 20  # Ensure batch size is consistent

DS_CONFIG = {
    "train_batch_size": BATCH_SIZE,  
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,  # ZeRO-2 for performance and memory optimization
        "offload_optimizer": {"device": "none"},  # Keep optimizer on GPU
        "offload_param": {"device": "none"}  # Keep parameters on GPU
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.00005,  # Learning rate
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.00005,  # Max LR after warmup
            "warmup_num_steps": 1000  # Warmup steps
        }
    },
}



# Load tokenizer and model
tokenizer = load_tokenizer("bert-base-uncased")
model = Model("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/contrastive/ds_step_18000")

# Load data with specified batch size
train_loader, test_loader = get_supervised_contrastive_dataloader('./data/hard_neg', tokenizer, batch_size=BATCH_SIZE)
criterion = SupervisedInfoNCELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=DS_CONFIG
)


step = 0
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1} (Training)")
    
    for batch in progress_bar:

        batch = {key: batch[key].view(-1, batch['input_ids'].shape[-1]).to(device) for key in batch.keys()}
        outputs = model.encode(batch).view(-1,7,768)
        # Assuming outputs.shape = (batch_size, num_neg + 2, 768)
        outputs1 = outputs[:, 0, :]   # First element: (batch_size, 768)
        outputs2 = outputs[:, 1, :]   # Second element: (batch_size, 768)
        negatives = outputs[:, 2:, :] # Remaining: (batch_size, num_neg, 768)


        loss = criterion(outputs1, outputs2, negatives)
        model.backward(loss)
        model.step()
        total_loss += loss.item()
        step += 1
        
        if step % SAVE_STEP == 0:
            print(f'Saving model at step {step}')
            model.module.save_pretrained(f'./weights/supervised_contrastive/ds_step_{step}/')
        
        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix({'Step': step, "Loss": avg_loss})
    
    print(f"Epoch {epoch + 1}, Step: {step}, Loss: {avg_loss}")
    
    model.eval()
    total_eval_loss = 0
    progress_bar = tqdm.tqdm(test_loader, desc=f"Epoch {epoch + 1} (Testing)")
    
    with torch.no_grad():
        for batch in progress_bar:
            batch = {key: batch[key].view(-1, batch['input_ids'].shape[-1]).to(device) for key in batch.keys()}
            outputs = model.encode(batch).view(-1,7,768)
            # Assuming outputs.shape = (batch_size, num_neg + 2, 768)
            outputs1 = outputs[:, 0, :]   # First element: (batch_size, 768)
            outputs2 = outputs[:, 1, :]   # Second element: (batch_size, 768)
            negatives = outputs[:, 2:, :] # Remaining: (batch_size, num_neg, 768)
            
            loss = criterion(outputs1, outputs2, negatives)
            total_eval_loss += loss.item()
            
            avg_eval_loss = total_eval_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({"Eval Loss": avg_eval_loss})
    
    print(f"Epoch {epoch + 1}, Evaluation Loss: {avg_eval_loss}")