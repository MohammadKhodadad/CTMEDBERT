from transformers import AutoModel, AutoModelForMaskedLM
import torch
import torch.nn as nn
import os

class Model(nn.Module):  # Inherit from torch.nn.Module
    def __init__(self, hf_address="bert-base-uncased", cache_dir='./cache',task='contrastive'):
        super(Model, self).__init__()
        self.hf_address = hf_address
        self.cache_dir = cache_dir
        if task=='contrastive':
            self.model = AutoModel.from_pretrained(hf_address, cache_dir=cache_dir)
        elif task=='mlm':
            self.model = AutoModelForMaskedLM.from_pretrained(hf_address, cache_dir=cache_dir)
        else:
            raise Exception('task has to be mlm or contastive.')


    def encode(self, inputs_, use_cls=True):
        # Forward pass to get the model outputs
        outputs = self.model(**inputs_)

        # Extract the token embeddings from the model's output
        token_embeddings = outputs.last_hidden_state
        if use_cls:
            cls_embedding = token_embeddings[:, 0, :]
            return cls_embedding
        else:

            attention_mask = inputs_['attention_mask'].unsqueeze(-1)
            masked_embeddings = token_embeddings * attention_mask
            mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
            return mean_embedding

    def save_weights(self, save_path):
        if  not os.path.exists('./weights') :
            os.makedirs('./weights')
        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights saved at {save_path}.")

    def load_weights(self, load_path):
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path))
            print(f"Model weights loaded from {load_path}.")
        else:
            print(f"Weight file {load_path} does not exist.")
    def save_pretrained(self, address):
        self.model.save_pretrained(address)

    def forward(self, **inputs_):
        outputs = self.model(**inputs_)
        return outputs
if __name__ == "__main__":
    model_instance = Model()  # The model will be cached in ./cache
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_instance = model_instance.to(device)
    from tokenizer_loader import load_tokenizer
    tokenizer = load_tokenizer()
    inputs = tokenizer("Medical BERT is a powerful tool.", return_tensors="pt", padding=True, truncation=True)
    print(inputs.keys())
    inputs={key:inputs[key].to(device) for key in inputs.keys()}
    embedding = model_instance.encode(inputs)
    print(f"Encoded embedding: {embedding}")
    model_instance.save_weights("./weights/saved_model_weights.pth")
    model_instance.load_weights("./weights/saved_model_weights.pth")
    another_inputs = tokenizer("Another test sentence.", return_tensors="pt", padding=True, truncation=True)
    another_inputs={key:another_inputs[key].to(device) for key in another_inputs.keys()}
    another_embedding = model_instance.encode(another_inputs)
    print(f"Another encoded embedding: {another_embedding}")
