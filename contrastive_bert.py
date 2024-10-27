import torch
import pandas as pd
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.loss import InfoNCELoss
from utils.data_loader import get_contrastive_dataloader
from utils.optimizer import get_optimizer_and_scheduler

EPOCHS=100
WARM_UP_STEPS= 1000
TOTAL_STEPS = 10000

tokenizer = load_tokenizer("bert-base-uncased")
model = Model("bert-base-uncased")
data=pd.read_csv("./data/data.csv")
data_loader = get_contrastive_dataloader(data, tokenizer)
criterion = InfoNCELoss()
optimizer, warmup_scheduler, cosine_scheduler = get_optimizer_and_scheduler(model,0.0001,WARM_UP_STEPS, TOTAL_STEPS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
step=0
for epoch in range(EPOCHS):
    model.train()
    for param_group in optimizer.param_groups:
        print(f"Epoch {epoch}: Learning Rate = {param_group['lr']}")
    for batch1, batch2 in data_loader:
        batch1 = {key: batch1[key].to(device) for key in batch1.keys()}
        batch2 = {key: batch2[key].to(device) for key in batch2.keys()}
        outputs1 = model.encode(batch1)
        outputs2 = model.encode(batch2)
        loss = criterion(outputs1, outputs2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        warmup_scheduler.step()
        if step >= WARM_UP_STEPS:
            cosine_scheduler.step()
        step+=1

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
