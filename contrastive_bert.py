import torch
import pandas as pd
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.loss import InfoNCELoss
from utils.data_loader import get_contrastive_dataloader

tokenizer = load_tokenizer("bert-base-uncased")
model = Model("bert-base-uncased")
data=pd.read_csv("./data/data.csv")
data_loader = get_contrastive_dataloader(data, tokenizer)
criterion = InfoNCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(100):
    model.train()
    for batch1, batch2 in data_loader:
        batch1 = {key: batch1[key].to(device) for key in batch1.keys()}
        batch2 = {key: batch2[key].to(device) for key in batch2.keys()}
        outputs1 = model.encode(batch1)
        outputs2 = model.encode(batch2)
        loss = criterion(outputs1, outputs2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
