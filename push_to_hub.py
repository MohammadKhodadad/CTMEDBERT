from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer, models
import torch

# Define paths
# hf_address = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/supervised_contrastive/ds_step_16000"
# tokenizer_name = "bert-base-uncased"

hf_address = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/contrastive_gte_11_ali/ds_step_25000"
tokenizer_name = "thenlper/gte-base"

model = AutoModel.from_pretrained(hf_address)
# checkpoint = torch.load("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/contrastive/ds_step_18000/global_step18000/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt", map_location="cpu")

# # Load config
# config = AutoConfig.from_pretrained("bert-base-uncased")  # Replace with your actual model

# # Initialize model
# model = AutoModel.from_config(config)
# print(checkpoint.keys())
# model.load_state_dict(checkpoint, strict=False) 

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
repo_name = "skyfury/CTMEDGTE-cl11-chem-step_25000"


model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)



# # Load the tokenizer and transformer model
# word_embedding_model = models.Transformer(repo_name, max_seq_length=256)

# # Create the pooling layer
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# # Initialize the SentenceTransformer model
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# # Push model to Hugging Face Hub
# # model.save("local_ctmedgte")  # Save locally before pushing (optional)
# model.push_to_hub('skyfury/CTMEDGTE7_encoder')

# print(f"Model pushed successfully to {repo_name}")


# # Step 1: Load the transformer model
# transformer = models.Transformer(repo_name)

# # Step 2: Define a pooling layer using the CLS token
# pooling = models.Pooling(
#     word_embedding_dimension=transformer.get_word_embedding_dimension(),
#     pooling_mode_mean_tokens=False,  # Disable mean pooling
#     pooling_mode_cls_token=True,     # Enable CLS token pooling
#     pooling_mode_max_tokens=False    # Disable max pooling
# )

# # Step 3: Combine transformer and pooling into a SentenceTransformer model
# sentence_transformer_model = SentenceTransformer(modules=[transformer, pooling])

# # Step 4: Name your model
# model_save_name = "skyfury/CTMEDGTE_CLS_Encoder2"


# sentence_transformer_model.push_to_hub(model_save_name)
# tokenizer.push_to_hub(model_save_name)