import torch
from mteb import MTEB
from mteb.tasks import AbsTaskReranking
from transformers import AutoTokenizer, AutoModel

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_path_or_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = AutoModel.from_pretrained(model_path_or_name)
    return model, tokenizer

# Custom task for reranking
class CustomRerankingTask(AbsTaskReranking):
    def __init__(self, data, model, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.description = "Custom reranking task for medical queries"
        self.data = data
        self.model = model
        self.tokenizer = tokenizer

    def load_data(self):
        # Format the data as needed by MTEB
        self.corpus = [{"text": doc} for query, doc, _ in self.data]
        self.queries = [{"text": query} for query, _, _ in self.data]
        self.relevant_docs = {i: [i] for i in range(len(self.data))}

    def encode_texts(self, texts):
        # Generate embeddings using the model and tokenizer
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

# Function to evaluate the model
def evaluate_model_for_retrieval(model, tokenizer, data):
    # Create and initialize the custom task
    custom_task = CustomRerankingTask(data=data, model=model, tokenizer=tokenizer)
    custom_task.load_data()

    # Initialize the MTEB benchmark and run the evaluation
    mteb = MTEB(tasks=[custom_task])
    results = mteb.run(models={"model": model, "tokenizer": tokenizer})
    print("Evaluation results:", results)

    
if __name__ == "__main__":
    # Load model and tokenizer
    model_path_or_name = "sentence-transformers/all-MiniLM-L6-v2"
    model, tokenizer = load_model_and_tokenizer(model_path_or_name)

    # Example data for the reranking task: (query, document, label)
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
