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
TOTAL_STEPS = 10000
SAVE_STEP= 1000

tokenizer = load_tokenizer("bert-base-uncased")
model = Model("bert-base-uncased")
create_cl_data_from_csv('./data/discharge_processed.csv','./data/','history_of_present_illness','chief_complaint')
data=pd.read_csv("./data/history_of_present_illness_vs_chief_complaint_cleaned.csv")
data_loader = get_contrastive_dataloader(data, tokenizer)
criterion = InfoNCELoss()
optimizer, scheduler = get_optimizer_and_scheduler(model,0.0001,WARM_UP_STEPS, TOTAL_STEPS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
step=0
for epoch in range(EPOCHS):
    model.train()
    for param_group in optimizer.param_groups:
        print(f"Epoch {epoch}: Learning Rate = {param_group['lr']}")
    for batch1, batch2 in tqdm.tqdm(data_loader):
        batch1 = {key: batch1[key].to(device) for key in batch1.keys()}
        batch2 = {key: batch2[key].to(device) for key in batch2.keys()}
        outputs1 = model.encode(batch1)
        outputs2 = model.encode(batch2)
        loss = criterion(outputs1, outputs2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step+=1
    if step>0 and step%SAVE_STEP==0:
        model.save_pretrained(f'./weights/contrastive/step_{step}/')

    

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
