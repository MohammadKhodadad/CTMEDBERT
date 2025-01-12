import torch
import tqdm
import pandas as pd
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.mlm_data_loader import get_mlm_dataloader
from utils.optimizer import get_optimizer_and_scheduler
from utils.dataloader.mimic import create_mimic_txt_from_csv
from utils.dataloader.pubmed import download_pubmed_mlm
from utils.dataloader.trialsgov import create_trials_mlm_data

# EPOCHS=1000
# WARM_UP_STEPS= 1000
# TOTAL_STEPS = 100000
# SAVE_STEP= 10000

# tokenizer = load_tokenizer("bert-base-uncased")
# model = Model("bert-base-uncased",task='mlm')
print('Handling Mimic Data')
create_mimic_txt_from_csv('./data/discharge_processed.csv','./data/txts')
print('Handling PubMed Data')
download_pubmed_mlm('./data/txts')
print('Handling Trials Data')
create_trials_mlm_data('./data/clinical_trials_all_studies.csv','./data/txts')
# save_abstracts_for_mlm(pubmed_abstracts,output_dir= 'created_data',name=query.replace(' ','_'))
# train_loader,test_loader = get_mlm_dataloader('./data', tokenizer)

# optimizer, scheduler = get_optimizer_and_scheduler(model,0.00005,WARM_UP_STEPS, TOTAL_STEPS)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# step=0
# for epoch in range(EPOCHS):
#     total_loss=0
#     model.train()
#     for param_group in optimizer.param_groups:
#         print(f"Epoch {epoch}: Learning Rate = {param_group['lr']}")
#     progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1} (Training)")
#     for batch in progress_bar:
#         batch = {key: batch[key].to(device) for key in batch.keys()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         total_loss+=loss.item()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         step+=1
#         if step>0 and step%SAVE_STEP==0:
#             print(f'Saving model in step: {step}')
#             model.save_pretrained(f'./weights/mlm/step_{step}/')
#         avg_loss = total_loss / (progress_bar.n + 1)
#         progress_bar.set_postfix({'Step':step,"Loss": avg_loss})
#     print(f"Epoch {epoch + 1}, Step: {step}, Loss: {avg_loss}")
#     model.eval()
#     total_eval_loss = 0
#     progress_bar = tqdm.tqdm(test_loader, desc=f"Epoch {epoch + 1} (Testing)")
    
#     with torch.no_grad():
#         for batch in progress_bar:
#             batch = {key: batch[key].to(device) for key in batch.keys()}
#             outputs = model(**batch)
#             loss = outputs.loss
#             total_eval_loss += loss.item()
            
#             avg_eval_loss = total_eval_loss / (progress_bar.n + 1)
#             progress_bar.set_postfix({"Eval Loss": avg_eval_loss})
    
#     print(f"Epoch {epoch + 1}, Evaluation Loss: {avg_eval_loss}")