import torch
import tqdm
import pandas as pd
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.mlm_data_loader import get_mlm_dataloader
from utils.optimizer import get_optimizer_and_scheduler
from utils.data import create_txt_from_csv

EPOCHS=100
WARM_UP_STEPS= 1000
TOTAL_STEPS = 10000
SAVE_STEP= 1000

tokenizer = load_tokenizer("bert-base-uncased")
model = Model("bert-base-uncased",task='mlm')
create_txt_from_csv('./data/discharge_processed.csv','./data/')
data_loader = get_mlm_dataloader('./data', tokenizer)

optimizer, warmup_scheduler, cosine_scheduler = get_optimizer_and_scheduler(model,0.0001,WARM_UP_STEPS, TOTAL_STEPS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
step=0
for epoch in range(EPOCHS):
    model.train()
    for param_group in optimizer.param_groups:
        print(f"Epoch {epoch}: Learning Rate = {param_group['lr']}")
    for batch in tqdm.tqdm(data_loader):
        batch = {key: batch[key].to(device) for key in batch.keys()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        warmup_scheduler.step()
        if step >= WARM_UP_STEPS:
            cosine_scheduler.step()
        step+=1
    if step>0 and step%SAVE_STEP==0:
        model.save_pretrained(f'./weights/mlm/step_{step}/')

    

#     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
