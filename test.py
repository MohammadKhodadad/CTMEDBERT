import torch
import tqdm
import deepspeed
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.loss import InfoNCELoss
from utils.contrastive_data_loader_ordered import get_contrastive_dataloader
from utils.dataloader.mimic import create_mimic_cl_data_from_csv
from utils.dataloader.pubmed import download_pubmed_cl
from utils.dataloader.trialsgov import create_trials_contrastive_learning_data
from utils.dataloader.medmcqa import create_medmcqa_contrastive_leanring_data
from utils.dataloader.medqa import create_medqa_contrastive_leanring_data
from utils.dataloader.medquad import create_medquad_contrastive_leanring_data
from utils.dataloader.wikipedia import create_wiki_cl
from utils.dataloader.curev1 import create_curev1_contrastive_learning_data
from utils.dataloader.biomedarxiv import create_medrxiv_sentence_data, create_biorxiv_sentence_data
from utils.dataloader.treccovid import create_trec_covid_contrastive_data
from utils.dataloader.nfcorpus import create_nfcorpus_contrastive_learning_data
from utils.optimizer import get_optimizer_and_scheduler



# print('Handling Wiki Data')
# create_wiki_cl()
# print('Handling Mimic Data')
# create_mimic_cl_data_from_csv('./data/discharge_processed.csv','./data/csvs','discharge_diagnosis',['chief_complaint','history_of_present_illness'])
# print('Handling PubMed Data')
download_pubmed_cl('./data/csvs')
# print('Handling Trials Data')
# create_trials_contrastive_learning_data('./data/clinical_trials_all_studies.csv','./data/csvs')
# print('Handling medmcqa Data')
# create_medmcqa_contrastive_leanring_data('./data/csvs')
# print('Handling medqa Data')
# create_medqa_contrastive_leanring_data('./data/csvs')
# print('Handling medquad Data')
# create_medquad_contrastive_leanring_data('./data/csvs')
# print('Handling curev1 Data')
# create_curev1_contrastive_learning_data('./data/csvs')
# print('Handling Biorxiv Data')
# create_biorxiv_sentence_data('./data/csvs')
# print('Handling Medrxiv Data')
# create_medrxiv_sentence_data('./data/csvs')
# print("Handling TREC data")
# create_trec_covid_contrastive_data('./data/csvs')
# print('Handling nfcorpus data')
# create_nfcorpus_contrastive_learning_data('./data/csvs')


# BATCH_SIZE=10
# tokenizer = load_tokenizer("thenlper/gte-base")
# train_loader, test_loader = get_contrastive_dataloader('./data/csvs', tokenizer, batch_size=BATCH_SIZE,max_length=512)
