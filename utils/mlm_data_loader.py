import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling

# Step 1: Load and tokenize the text files
class TokenizedChunkedDataset:
    def __init__(self, directory_path, tokenizer, chunk_size=128):
        # Get all the files from the directory
        self.files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

        # Step 1: Load and concatenate all text files
        self.full_text = self.load_all_text_files()
        print(f"full_text: {len(self.full_text)}")
        self.full_text = self.full_text [:10000000]
        print(f"full_text: {len(self.full_text)}")
        # Step 2: Tokenize the full concatenated text without truncation
        self.tokenized_data = self.tokenize_full_text()
        print(f"tokenized_data: {len(self.tokenized_data)}")
        # Step 3: Calculate the number of chunks based on the chunk size
        self.num_chunks = len(self.tokenized_data['input_ids'][0]) // self.chunk_size
        print(f"THE NUMBER OF CHUNKS: {self.num_chunks}")
    def load_all_text_files(self):
        full_text = ""
        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                full_text += f.read() + " "  # Concatenate all the text files into a single text
        return full_text

    def tokenize_full_text(self):
        # Tokenize the full concatenated text WITHOUT truncation
        tokenized_data = self.tokenizer(self.full_text, return_tensors="pt", padding=False, truncation=False)
        return tokenized_data

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        # Step 4: Retrieve a specific chunk based on idx and chunk_size
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        # Get the chunk of input_ids, attention_mask, and token_type_ids
        input_ids_chunk = self.tokenized_data['input_ids'][0][start_idx:end_idx]
        attention_mask_chunk = torch.ones_like(input_ids_chunk)  # Create attention mask for the chunk

        # Handle token_type_ids if present
        token_type_ids_chunk = None
        if 'token_type_ids' in self.tokenized_data:
            token_type_ids_chunk = self.tokenized_data['token_type_ids'][0][start_idx:end_idx]
        
        return {
            'input_ids': input_ids_chunk,
            'attention_mask': attention_mask_chunk,
            'token_type_ids': token_type_ids_chunk if token_type_ids_chunk is not None else torch.zeros_like(input_ids_chunk)
        }


# Step 2: DataLoader function for efficient retrieval
def get_mlm_dataloader(directory_path, tokenizer, batch_size=32, max_length=128,mlm_probability=0.15):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )
    dataset = TokenizedChunkedDataset(directory_path, tokenizer=tokenizer, chunk_size=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=data_collator)


# Step 3: Example usage
if __name__ == "__main__":
    # Directory containing .txt files
    directory_path = "../data"
    
    # Load the tokenizer
    from tokenizer_loader import load_tokenizer
    tokenizer=load_tokenizer()
    # Load the DataLoader
    tokenized_dataloader = get_mlm_dataloader(directory_path, tokenizer)

    # Iterate through the DataLoader and inspect the batches
    for batch in tokenized_dataloader:
        print("Input IDs:", batch['input_ids'][0])
        print(tokenizer.decode(batch['input_ids'][0]))
        print("Input IDs:", batch['labels'][0])
        print(tokenizer.decode(batch['labels'][0][batch['labels'][0]>=0]))
        # print("Attention Mask:", batch['attention_mask'][0])
        # print("Token type ids:", batch['token_type_ids'][0])
        
