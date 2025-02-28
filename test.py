
# from utils.contrastive_data_loader import get_contrastive_dataloader
# from utils.dataloader.mimic import create_mimic_cl_data_from_csv
# from utils.dataloader.pubmed import download_pubmed_cl
# from utils.dataloader.trialsgov import create_trials_contrastive_learning_data
# from utils.dataloader.medmcqa import create_medmcqa_contrastive_leanring_data
# from utils.dataloader.medqa import create_medqa_contrastive_leanring_data
# from utils.dataloader.medquad import create_medquad_contrastive_leanring_data


# # from utils.optimizer import get_optimizer_and_scheduler




# # print('Handling Mimic Data')
# # create_mimic_cl_data_from_csv('./data/discharge_processed.csv','./data/csvs','discharge_diagnosis',['chief_complaint','history_of_present_illness'])
# print('Handling PubMed Data')
# download_pubmed_cl('./data/csvs')
# # print('Handling Trials Data')
# # create_trials_contrastive_learning_data('./data/clinical_trials_all_studies.csv','./data/csvs')
# # print('Handling medmcqa Data')
# # create_medmcqa_contrastive_leanring_data('./data/csvs')
# # print('Handling medqa Data')
# # create_medqa_contrastive_leanring_data('./data/csvs')
# # print('Handling medquad Data')
# # create_medquad_contrastive_leanring_data('./data/csvs')

import torch
import tqdm
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.loss import InfoNCELoss, AnchorChunkInfoNCELoss, InfoNCELossChunked
from utils.contrastive_data_loader import get_contrastive_dataloader
from utils.dataloader.mimic import create_mimic_cl_data_from_csv
from utils.dataloader.pubmed import download_pubmed_cl
from utils.dataloader.trialsgov import create_trials_contrastive_learning_data
from utils.dataloader.medmcqa import create_medmcqa_contrastive_leanring_data
from utils.dataloader.medqa import create_medqa_contrastive_leanring_data
from utils.dataloader.medquad import create_medquad_contrastive_leanring_data
from utils.dataloader.wikipedia import create_wiki_cl


tokenizer = load_tokenizer("Alibaba-NLP/gte-base-en-v1.5")
model = Model("Alibaba-NLP/gte-base-en-v1.5")