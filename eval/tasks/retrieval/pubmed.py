from __future__ import annotations
from typing import Any
import datasets
import json
import pandas as pd
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
class PubMedTitleAbsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PubMedTitleAbsRetrieval",
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

      with open("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/eval/dev/medical_data.json", "r", encoding="utf-8") as file:
          data = json.load(file)
      self.corpus = {
          "test": {}}
      self.queries = {
          "test": {}}
      self.relevant_docs = {
          "test": {}}

      for i, article in enumerate(data):
        # Extract the title and abstract
        try:
          title = article.get("title", "").strip()
          abstract = article.get("abstract", "").strip()
        except:
          pass
        # Fill in the corpus, queries, and relevant docs
        if title and abstract:
          self.corpus["test"][f"doc_{i}"] = {"_id": f"d{i}", "title": title, "text": abstract}
          self.queries["test"][f"q_{i}"] = title
          self.relevant_docs["test"][f"q_{i}"] = {f"doc_{i}": 1}
      self.data_loaded = True

    def dataset_transform(self):
      pass
      self.data_loaded = True

    def dataset_transform(self):
      pass

if __name__ == "__main__":
    task = PubMedTitleAbsRetrieval()
    mteb = MTEB(tasks=[task])
    # model = get_model("bert-base-uncased")
    # model = get_model("sentence-transformers/all-MiniLM-L6-v2")
    # model = get_model('intfloat/e5-base')
    # model = get_model("medicalai/ClinicalBERT")
    # model = get_model("nomic-ai/nomic-embed-text-v1")
    model = get_model("emilyalsentzer/Bio_ClinicalBERT")
    # model = get_model('kamalkraj/BioSimCSE-BioLinkBERT-BASE')
    # model = get_model('malteos/scincl')
    
    # model = HuggingFaceEmbedder('nomic-ai/nomic-embed-text-v1')
    # model = HuggingFaceEmbedder('intfloat/e5-base')
    # transformer= models.Transformer('intfloat/e5-base-v2')
    # pooling_model = models.Pooling(
    #     transformer.get_word_embedding_dimension(),
    #     pooling_mode_cls_token=False,  # Use CLS token for pooling
    #     pooling_mode_mean_tokens=True,
    #     pooling_mode_max_tokens=False
    # )

    # transformer= models.Transformer('/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/contrastive/step_45000',tokenizer_name_or_path ='bert-base-uncased')
    # pooling_model = models.Pooling(
    #     transformer.get_word_embedding_dimension(),
    #     pooling_mode_cls_token=True,  # Use CLS token for pooling
    #     pooling_mode_mean_tokens=False,
    #     pooling_mode_max_tokens=False
    # )
    # model = SentenceTransformer(modules=[transformer, pooling_model])
    results = mteb.run(model)
    print("Evaluation results:", results)
    # for task in results:
    #   print(task)