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

for epoch in range(10):
    model.train()
    for batch in data_loader:
        batch={key:batch[key].to(device) for key in batch.keys()}
        input_ids1 = batch['input_ids1']
        attention_mask1 = batch['attention_mask1']
        token_type_ids1 = batch['token_type_ids1']
        batch1={'input_ids':input_ids1,'attention_mask':attention_mask1,'token_type_ids':token_type_ids1}
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        token_type_ids2 = batch['token_type_ids2'].to(device)
        batch2={'input_ids':input_ids2,'attention_mask':attention_mask2,'token_type_ids':token_type_ids2}
        outputs1 = model.encode(batch1)
        outputs2 = model.encode(batch2)
        loss = criterion(outputs1, outputs2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
