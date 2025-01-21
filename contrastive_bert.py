import torch
import tqdm
import pandas as pd
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

tokenizer = load_tokenizer("bert-base-uncased")
# model = Model("bert-base-uncased")
model = Model("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/mlm/step_130000")
# print('Handling Mimic Data')
# create_mimic_cl_data_from_csv('./data/discharge_processed.csv','./data/csvs','discharge_diagnosis',['chief_complaint','history_of_present_illness'])
# print('Handling PubMed Data')
# download_pubmed_cl('./data/csvs')
# print('Handling Trials Data')
# create_trials_contrastive_learning_data('./data/clinical_trials_all_studies.csv','./data/csvs')

train_loader, test_loader = get_contrastive_dataloader('./data/csvs', tokenizer)
criterion = InfoNCELoss()
optimizer, scheduler = get_optimizer_and_scheduler(model,0.00005,WARM_UP_STEPS, TOTAL_STEPS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
step=0
for epoch in range(EPOCHS):
    total_loss=0
    model.train()
    for param_group in optimizer.param_groups:
        print(f"Epoch {epoch}: Learning Rate = {param_group['lr']}")
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1} (Training)")
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
            print(f'Saving model in step: {step}')
            model.save_pretrained(f'./weights/contrastive/step_{step}/')
        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix({'Step':step,"Loss": avg_loss})
    print(f"Epoch {epoch + 1}, Step: {step}, Loss: {avg_loss}")
    model.eval()
    total_eval_loss = 0
    progress_bar = tqdm.tqdm(test_loader, desc=f"Epoch {epoch + 1} (Testing)")
    
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

