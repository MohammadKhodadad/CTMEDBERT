from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models

# Define paths
hf_address = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/mlm/step_60000"
tokenizer_name = "bert-base-uncased"
model = AutoModel.from_pretrained(hf_address)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
repo_name = "skyfury/CTMEDBERT-step60000"


model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)


# Step 1: Load the transformer model
model_name = "skyfury/CTMEDBERT-step60000"
transformer = models.Transformer(model_name)

# Step 2: Define a pooling layer using the CLS token
pooling = models.Pooling(
    word_embedding_dimension=transformer.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=False,  # Disable mean pooling
    pooling_mode_cls_token=True,     # Enable CLS token pooling
    pooling_mode_max_tokens=False    # Disable max pooling
)

# Step 3: Combine transformer and pooling into a SentenceTransformer model
sentence_transformer_model = SentenceTransformer(modules=[transformer, pooling])

# Step 4: Name your model
model_save_name = "skyfury/CTMEDBERT_CLS_Encoder"