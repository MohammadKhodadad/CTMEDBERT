from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer, models
import torch

# Define paths
hf_address = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/contrastive/ds_step_18000"
tokenizer_name = "bert-base-uncased"
model = AutoModel.from_pretrained(hf_address)
# checkpoint = torch.load("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/contrastive/ds_step_18000/global_step18000/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt", map_location="cpu")

# # Load config
# config = AutoConfig.from_pretrained("bert-base-uncased")  # Replace with your actual model

# # Initialize model
# model = AutoModel.from_config(config)
# print(checkpoint.keys())
# model.load_state_dict(checkpoint, strict=False) 

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
repo_name = "skyfury/CTMEDBERT-cl3-step_18000"


model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)


# Step 1: Load the transformer model
transformer = models.Transformer(repo_name)

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
model_save_name = "skyfury/CTMEDBERT_CLS_Encoder3"


sentence_transformer_model.push_to_hub(model_save_name)
tokenizer.push_to_hub(model_save_name)