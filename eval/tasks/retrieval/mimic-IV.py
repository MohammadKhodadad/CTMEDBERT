from __future__ import annotations
from typing import Any
import datasets
from datasets import Dataset
from datasets import Dataset, DatasetDict
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd


from mteb import MTEB, get_model
from transformers import AutoTokenizer, AutoModel

from sentence_transformers import SentenceTransformer, models

class MIMICIVRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MIMICIVRetrieval",
        dataset={"path": "",
                 "revision": " ",

        },
        description="",
        reference="https://errors.pydantic.dev/2.9/v/url_parsing",
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""""",
        descriptive_stats={}
    )




    def load_data(self, **kwargs: Any) -> None:
      """Load dataset from HuggingFace hub or use sample data"""
      if self.data_loaded:
          return
      data=pd.read_csv('test_retrieval.csv')
      data = pd.read_csv('/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/chief_complaint_vs_history_of_present_illness_cleaned.csv')
      data=data.dropna()
      data=data.sample(500)
      self.corpus = {
          "test": {}}
      self.queries = {
          "test": {}}
      self.relevant_docs = {
          "test": {}}


      for i in range(data.shape[0]):
        row=data.iloc[i]
        self.corpus['test'][f'doc_{i}']={"_id": f"d{i}", "title": "", "text": row['sentence2']}
        self.queries['test'][f'q_{i}']=row['sentence1']
        self.relevant_docs['test'][f'q_{i}']={f"doc_{i}":1}

      self.data_loaded = True

    def dataset_transform(self):
      pass
      self.data_loaded = True

    def dataset_transform(self):
      pass



# class HuggingFaceEmbedder:
#     def __init__(self, model_address):
#         self.task_name='test1'
#         self.model_name='test1'
#         self.tokenizer = AutoTokenizer.from_pretrained(model_address)
#         self.model = AutoModel.from_pretrained(model_address)

#     def encode(self, text):

#         if isinstance(text, str):
#             text = [text]

#         # Tokenize the input text and convert to tensor
#         inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

#         # Generate embeddings using the model
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         # Use the last hidden state as the embeddings
#         embeddings = outputs.last_hidden_state.mean(dim=1)
#         return embeddings

# Instantiate the class and load data
if __name__ == "__main__":
    task = MIMICIVRetrieval()
    mteb = MTEB(tasks=[task])
    # model = get_model("bert-base-uncased")
    # model = get_model("sentence-transformers/all-MiniLM-L6-v2")
    # model = get_model('intfloat/e5-base')
    # model = get_model("medicalai/ClinicalBERT")
    # model = get_model("nomic-ai/nomic-embed-text-v1")
    # model = get_model("emilyalsentzer/Bio_ClinicalBERT")
    # model = get_model('kamalkraj/BioSimCSE-BioLinkBERT-BASE')
    # model = get_model('malteos/scincl')
    
    # model = HuggingFaceEmbedder('nomic-ai/nomic-embed-text-v1')
    # model = HuggingFaceEmbedder('intfloat/e5-base')
    # model.encode(['Hi!'],task_name='ScalaClassification')
    transformer= models.Transformer('/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/contrastive/step_45000',tokenizer_name_or_path ='bert-base-uncased')
    # transformer= models.Transformer('intfloat/e5-base-v2')
    pooling_model = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,  # Use CLS token for pooling
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False
    )

    # pooling_model = models.Pooling(
    #     transformer.get_word_embedding_dimension(),
    #     pooling_mode_cls_token=False,  # Use CLS token for pooling
    #     pooling_mode_mean_tokens=True,
    #     pooling_mode_max_tokens=False
    # )
    # # Create a SentenceTransformer instance
    model = SentenceTransformer(modules=[transformer, pooling_model])

    results = mteb.run(model)
    print("Evaluation results:", results)
    for task in results:
      print(task)