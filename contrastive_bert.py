import torch
import tqdm
import pandas as pd
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.loss import InfoNCELoss
from utils.contrastive_data_loader import get_contrastive_dataloader
from utils.optimizer import get_optimizer_and_scheduler
from utils.data import create_cl_data_from_csv

EPOCHS=100
WARM_UP_STEPS= 1000
TOTAL_STEPS = 1000000
SAVE_STEP= 10000

tokenizer = load_tokenizer("bert-base-uncased")
model = Model("bert-base-uncased")
create_cl_data_from_csv('./data/discharge_processed.csv','./data/','history_of_present_illness','chief_complaint')
data=pd.read_csv("./data/history_of_present_illness_vs_chief_complaint_cleaned.csv")
train_dataset, test_dataset = get_contrastive_dataloader(data, tokenizer)
criterion = InfoNCELoss()
optimizer, scheduler = get_optimizer_and_scheduler(model,0.0001,WARM_UP_STEPS, TOTAL_STEPS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
step=0
for epoch in range(EPOCHS):
    total_loss=0
    model.train()
    for param_group in optimizer.param_groups:
        print(f"Epoch {epoch}: Learning Rate = {param_group['lr']}")
        progress_bar = tqdm.tqdm(train_dataset, desc=f"Epoch {epoch + 1}")
    for batch1, batch2 in progress_bar:
        batch1 = {key: batch1[key].to(device) for key in batch1.keys()}
        batch2 = {key: batch2[key].to(device) for key in batch2.keys()}
        outputs1 = model.encode(batch1)
        outputs2 = model.encode(batch2)
        loss = criterion(outputs1, outputs2)
        total_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step+=1
        if step>0 and step%SAVE_STEP==0:
            print('Saving model in step: {step}')
            model.save_pretrained(f'./weights/contrastive/step_{step}/')
        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix({'Step':step,"Loss": avg_loss})
    print(f"Epoch {epoch + 1}, Step: {step}, Loss: {avg_loss}")
    model.eval()
    total_eval_loss = 0
    progress_bar = tqdm.tqdm(test_dataset, desc=f"Epoch {epoch + 1} (Testing)")
    
    with torch.no_grad():
        for batch1, batch2 in progress_bar:
            # Move batches to the device
            batch1 = {key: batch1[key].to(device) for key in batch1.keys()}
            batch2 = {key: batch2[key].to(device) for key in batch2.keys()}
            
            # Forward pass: encode both batches
            outputs1 = model.encode(batch1)
            outputs2 = model.encode(batch2)
            
            # Compute contrastive loss
            loss = criterion(outputs1, outputs2)
            total_eval_loss += loss.item()
            
            avg_eval_loss = total_eval_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({"Eval Loss": avg_eval_loss})
    
    print(f"Epoch {epoch + 1}, Evaluation Loss: {avg_eval_loss}")

