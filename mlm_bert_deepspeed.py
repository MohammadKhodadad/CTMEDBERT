import torch
import tqdm
import deepspeed
import pandas as pd
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.mlm_data_loader import get_mlm_dataloader
from utils.optimizer import get_optimizer_and_scheduler

deepspeed.init_distributed(dist_backend='nccl')

# Hyperparameters
EPOCHS = 1000
WARM_UP_STEPS = 1000
TOTAL_STEPS = 200000
SAVE_STEP = 10000
BATCH_SIZE = 32
LEARNING_RATE = 5e-5

# Load tokenizer and model
tokenizer = load_tokenizer("bert-base-uncased")
model = Model("bert-base-uncased", task='mlm')

# Load dataset
train_loader, test_loader = get_mlm_dataloader('./data/txts', tokenizer, batch_size=BATCH_SIZE, distributed=True)

# Configure optimizer and scheduler
optimizer, scheduler = get_optimizer_and_scheduler(model, base_lr=LEARNING_RATE, warmup_steps=WARM_UP_STEPS, total_steps=TOTAL_STEPS)

# DeepSpeed configuration (NO Gradient Accumulation, No ZeRO)
ds_config = {
    "train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": 1,  # No gradient accumulation
    "fp16": {"enabled": False},  # Disable FP16 (Enable if needed)
    "zero_optimization": {"stage": 0},  # No ZeRO optimization
}

# Initialize DeepSpeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    # lr_scheduler = scheduler,
    model_parameters=model.parameters(),
    config=ds_config
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
step = 0

# Training Loop
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    
    for param_group in optimizer.param_groups:
        print(f"Epoch {epoch}: Learning Rate = {param_group['lr']}")
    
    progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1} (Training)") # disable=(model.local_rank  != 0)
    
    for batch in progress_bar:
        batch = {key: batch[key].to(device) for key in batch.keys()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        
        model.backward(loss)
        model.step()
        scheduler.step()

        step += 1
        total_loss += loss.item()
        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix({'Step': step, "Loss": avg_loss})
        if step%200==0:
            print(f"Step: {step}", flush=True)
        if step > 0 and step % SAVE_STEP == 0 and model.local_rank == 0:
            print(f'Saving model at step: {step}')
            model.save_checkpoint(f'./weights/mlm/step_{step}/')
    
    print(f"Epoch {epoch + 1}, Step: {step}, Loss: {avg_loss}")
    
    # Evaluation Loop
    model.eval()
    total_eval_loss = 0
    progress_bar = tqdm.tqdm(test_loader, desc=f"Epoch {epoch + 1} (Testing)")
    
    with torch.no_grad():
        for batch in progress_bar:
            batch = {key: batch[key].to(device) for key in batch.keys()}
            outputs = model(**batch)
            loss = outputs.loss
            
            total_eval_loss += loss.item()
            avg_eval_loss = total_eval_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({"Eval Loss": avg_eval_loss})
    
    print(f"Epoch {epoch + 1}, Evaluation Loss: {avg_eval_loss}")
