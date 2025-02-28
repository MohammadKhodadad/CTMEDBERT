import torch
import tqdm
import pandas as pd
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.loss import SupervisedInfoNCELoss
from utils.supervised_contrastive_data_loader import get_supervised_contrastive_dataloader
from utils.optimizer import get_optimizer_and_scheduler
from utils.dataloader.get_negatives import save_hard_negatives, load_and_process_hard_negatives

# directory = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/csvs"  # Change this to your CSV directory
# embedding_file = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/embeddings.npy"
# output_file = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/hard_neg/hard_negative_v1.csv"

# save_hard_negatives(directory, embedding_file=embedding_file)
# load_and_process_hard_negatives(directory, embedding_file=embedding_file, output_file=output_file)





EPOCHS=100
WARM_UP_STEPS= 1000
TOTAL_STEPS = 100000
SAVE_STEP= 5000

tokenizer = load_tokenizer("bert-base-uncased")

model = Model("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/contrastive/ds_step_18000")

train_loader, test_loader = get_supervised_contrastive_dataloader('./data/hard_neg', tokenizer)
criterion = SupervisedInfoNCELoss()
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
    for batch1, batch2, hard_negatives in progress_bar:
        batch1 = {key: batch1[key].to(device) for key in batch1.keys()}
        batch2 = {key: batch2[key].to(device) for key in batch2.keys()}
        hard_negatives = {key: hard_negatives[key].view(-1, hard_negatives['input_ids'].shape[-1]).to(device) for key in hard_negatives.keys()}
        outputs1 = model.encode(batch1)
        outputs2 = model.encode(batch2)
        negatives = model.encode(hard_negatives).view(-1,5,768)
        loss = criterion(outputs1, outputs2, negatives)
        total_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step+=1
        if step>0 and step%SAVE_STEP==0:
            print(f'Saving model in step: {step}')
            model.save_pretrained(f'./weights/supervised_contrastive/step_{step}/')
        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix({'Step':step,"Loss": avg_loss})
    print(f"Epoch {epoch + 1}, Step: {step}, Loss: {avg_loss}")
    model.eval()
    total_eval_loss = 0
    progress_bar = tqdm.tqdm(test_loader, desc=f"Epoch {epoch + 1} (Testing)")
    
    with torch.no_grad():
        for batch1, batch2, hard_negatives in progress_bar:
            # Move batches to the device
            batch1 = {key: batch1[key].to(device) for key in batch1.keys()}
            batch2 = {key: batch2[key].to(device) for key in batch2.keys()}
            hard_negatives = {key: batch2[key].view(-1, batch['hard_neg_ids'].shape[-1]).to(device) for key in batch2.keys()}

            # Forward pass: encode both batches
            outputs1 = model.encode(batch1)
            outputs2 = model.encode(batch2)
            negatives = model.encode(hard_negatives).view(-1,5,768)

            # Compute contrastive loss
            loss = criterion(outputs1, outputs2, negatives)
            total_eval_loss += loss.item()
            
            avg_eval_loss = total_eval_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({"Eval Loss": avg_eval_loss})
    
    print(f"Epoch {epoch + 1}, Evaluation Loss: {avg_eval_loss}")

