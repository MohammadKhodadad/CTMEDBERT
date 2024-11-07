import torch
from mteb import MTEB
from mteb.tasks import CustomTask
from transformers import AutoTokenizer, AutoModel


def load_model_and_tokenizer(model_path_or_name='bert-base-uncased', tokenizer_path_or_name='bert-base-uncased'):
    """
    Loads the model and tokenizer from a given path or model name.
    
    Args:
        model_path_or_name (str): Path to the model directory or model name from Hugging Face.
    
    Returns:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)
    model = AutoModel.from_pretrained(model_path_or_name)
    return model, tokenizer


def evaluate_model_for_retrieval(model, tokenizer, data, pooling_type='mean'):
    """
    Evaluates a given model and tokenizer on a retrieval task using MTEB, with support for mean pooling or CLS pooling.
    
    Args:
        model: The Hugging Face model to use for embeddings.
        tokenizer: The Hugging Face tokenizer to use for embeddings.
        data: Custom data for the retrieval task, formatted as a list of (query, document, label) tuples.
              - Example: [("query1", "document1", label1), ("query2", "document2", label2), ...]
        pooling_type (str): The type of pooling to use for sentence embeddings ('mean' or 'cls').
    
    Returns:
        None. Runs the MTEB evaluation and prints the results.
    """
    # Define the embedding function
    def embed_texts(texts):
        # Tokenize input texts
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Apply the selected pooling method
        if pooling_type == 'mean':
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        elif pooling_type == 'cls':
            # CLS pooling: use the embeddings of the [CLS] token
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError("Invalid pooling_type. Choose either 'mean' or 'cls'.")
        
        return embeddings.numpy()

    # Create a custom task class for retrieval
    class MyRetrievalTask(CustomTask):
        def __init__(self, data):
            super().__init__()
            self.data = data
        
        def load_data(self):
            return self.data

    # Initialize the custom retrieval task
    my_task = MyRetrievalTask(data)

    # Initialize the MTEB benchmark with the retrieval task
    benchmark = MTEB(tasks=[my_task])

    # Run the benchmark using the embedding function
    results= benchmark.run(embed_texts)
    print(results)




if __name__ == "__main__":
    # Load model and tokenizer
    model_path_or_name = "sentence-transformers/all-MiniLM-L6-v2"  # Or a directory path to your model
    model, tokenizer = load_model_and_tokenizer(model_path_or_name,model_path_or_name)

    # Example data for a retrieval task: (query, document, label)
    my_data = [
    ("What is diabetes?", "Diabetes is a chronic condition that affects how the body processes blood sugar.", 1),
    ("What are the symptoms of a heart attack?", "Symptoms of a heart attack include chest pain, shortness of breath, and nausea.", 1),
    ("How does insulin work?", "Insulin helps regulate blood sugar levels by facilitating the uptake of glucose into cells.", 1),
    ("What is hypertension?", "Hypertension, or high blood pressure, is a condition where the force of the blood against the artery walls is too high.", 1),
    ("What causes anemia?", "Anemia can be caused by a lack of iron, vitamin deficiency, chronic diseases, or blood loss.", 1),
    ("What is asthma?", "Asthma is a chronic respiratory condition characterized by airway inflammation and constriction.", 1),
    ("What are the side effects of ibuprofen?", "Common side effects of ibuprofen include stomach pain, nausea, and dizziness.", 1),
    ("How is cancer treated?", "Cancer treatment options include chemotherapy, radiation therapy, and surgery.", 1),
    ("What is pneumonia?", "Pneumonia is a lung infection that causes inflammation and fluid accumulation in the air sacs.", 1),
    ("What is osteoporosis?", "Osteoporosis is a bone disease that occurs when the body loses too much bone or makes too little bone.", 1),
    ("What causes migraine headaches?", "Migraines can be triggered by stress, certain foods, hormonal changes, and lack of sleep.", 1),
    ("What is rheumatoid arthritis?", "Rheumatoid arthritis is an autoimmune disease that causes inflammation in the joints.", 1),
    ("What is the function of the liver?", "The liver detoxifies chemicals, metabolizes drugs, and produces bile for digestion.", 1),
    ("How does HIV affect the body?", "HIV attacks the immune system, specifically CD4 cells, weakening the body's defense against infections.", 1),
    ("What is a stroke?", "A stroke occurs when blood flow to a part of the brain is interrupted, causing brain cells to die.", 1),
    ("What are the symptoms of depression?", "Symptoms of depression include persistent sadness, loss of interest, fatigue, and changes in appetite.", 1),
    ("What is COPD?", "COPD, or chronic obstructive pulmonary disease, is a progressive lung disease that causes breathing difficulties.", 1),
    ("How is tuberculosis transmitted?", "Tuberculosis is transmitted through the air when an infected person coughs or sneezes.", 1),
    ("What is an EKG?", "An EKG, or electrocardiogram, measures the electrical activity of the heart.", 1),
    ("What is a hernia?", "A hernia occurs when an organ pushes through an opening in the muscle or tissue holding it in place.", 1),
    ("What is hypothyroidism?", "Hypothyroidism is a condition where the thyroid gland does not produce enough thyroid hormone.", 1),
    ("What is a kidney stone?", "A kidney stone is a hard deposit made of minerals and salts that forms inside the kidneys.", 1),
    ("What causes a urinary tract infection?", "A urinary tract infection is usually caused by bacteria entering the urinary tract.", 1),
    ("What is sepsis?", "Sepsis is a life-threatening condition that occurs when the body's response to an infection causes tissue damage.", 1),
    ("What is anemia?", "Anemia is a condition characterized by a deficiency of red blood cells or hemoglobin in the blood.", 1),
    ]

    # Evaluate the model on the retrieval task
    evaluate_model_for_retrieval(model, tokenizer, my_data)
