import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


# Step 2: Dataset Class for Contrastive Learning
class ContrastiveDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the sentence pair
        sentence1 = self.data.iloc[index]['sentence1']
        sentence2 = self.data.iloc[index]['sentence2']

        # Tokenize the sentence pair
        inputs1 = self.tokenizer(sentence1, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True)
        inputs2 = self.tokenizer(sentence2, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True)

        # Return the tokenized inputs for contrastive learning
        return {
            'input_ids1': inputs1['input_ids'].squeeze(),  # Squeeze to remove extra dimension
            'attention_mask1': inputs1['attention_mask'].squeeze(),
            'token_type_ids1': inputs1['token_type_ids'].squeeze(),
            'input_ids2': inputs2['input_ids'].squeeze(),
            'attention_mask2': inputs2['attention_mask'].squeeze(),
            'token_type_ids2': inputs2['token_type_ids'].squeeze(),
        }

# Step 3: DataLoader Function
def get_contrastive_dataloader(dataframe,tokenizer, batch_size=2, max_length=128):
    dataset = ContrastiveDataset(dataframe, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 4: Test the DataLoader
if __name__ == "__main__":
    # Get the DataLoader
    # Step 1: Sample DataFrame (simulating the 2-column CSV file)
    data = {'sentence1': ["This is a positive sentence.", "This is another sentence."],
            'sentence2': ["This is a similar positive sentence.", "This is a dissimilar sentence."]}
    df = pd.DataFrame(data)
    from tokenizer_loader import load_tokenizer
    tokenizer=load_tokenizer()

    contrastive_dataloader = get_contrastive_dataloader(df,tokenizer)

    # Iterate through the DataLoader
    for batch in contrastive_dataloader:
        print("Input IDs Sentence 1:", batch['input_ids1'])
        print("Attention Mask Sentence 1:", batch['attention_mask1'])
        print("Input IDs Sentence 2:", batch['input_ids2'])
        print("Attention Mask Sentence 2:", batch['attention_mask2'])
