from transformers import AutoModel, AutoTokenizer
model_name="thenlper/gte-base"
model = AutoModel.from_pretrained(model_name).to('cpu').eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)