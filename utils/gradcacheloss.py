from contextlib import nullcontext

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F

# from contrastors.distributed import gather, gather_with_grad
import torch
from torch.utils.checkpoint import get_device_states, set_device_states
from transformers import AutoTokenizer
from transformers import AutoModel

# taken from: https://github.com/luyug/GradCache/blob/0c33638cb27c2519ad09c476824d550589a8ec38/src/grad_cache/context_managers.py
class RandContext:
    def __init__(self, tensors):
        if isinstance(tensors, dict):
            tensors = list(tensors.values())
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None

def clip_loss(
    query,
    document,
    logit_scale,
    step=None,
    # gather_enabled=False,
    tracker=None,
    dataset="",
    bidirectional=False,
):
    """Calculates the InfoNCE Loss for a batch of queries and documents.
    Inspired by: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py#L66

    Assumes that query.shape[0] <= document.shape[0]
    This will work for non-square matrices as well

    params:
        query: torch.Tensor of shape N x D
        document: torch.Tensor of shape M x D where M >= N
        temperature: torch.Tensor of shape 1

    returns:
        torch.Tensor of shape 1 corresponding to the loss
    """
    # if gather_enabled:
    #     document = gather_with_grad(document)

    device = query.device

    if query.dtype != document.dtype:
        document = document.to(query.dtype)

    labels = torch.arange(query.shape[0]).to(device)
    print(query.shape, document.shape)
    similarity_query_document = logit_scale(torch.matmul(query, document.T))
    num_logits = similarity_query_document.size(0)
    rank = dist.get_rank() if dist.is_initialized() else 0
    # calculate sub-batch labels
    labels = labels + rank * num_logits

    # if training with negatives
    # multiply by world size since we only gather the document embeddings
    labels = labels * (document.size(0) // (query.size(0) * 1))

    if bidirectional:
        similarity_document_query = logit_scale(torch.matmul(document, query.T))
        loss = (
            F.cross_entropy(similarity_query_document, labels) + F.cross_entropy(similarity_document_query, labels)
        ) * 1
    else:
        loss = F.cross_entropy(similarity_query_document, labels) * 1

    if tracker is not None:
        # this will only calculate 1/N accuracy where N is the number of gpus
        accuracy = (similarity_query_document.argmax(dim=1) == labels).float().mean()
        tracker.log({f"accuracy_{dataset}": accuracy.detach().cpu().item()}, step=step)

    return loss


def get_chunked_embeddings(model, chunks):
    embeddings = []
    rand_states = []

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            for chunk in chunks:
                rand_states.append(RandContext(chunk))
                emb = model(**chunk)
                embeddings.append(emb["last_hidden_state"].mean(axis=1))

    return torch.concat(embeddings, dim=0), rand_states


def accumulate_gradients(model, inputs, cache, rand_states):
    length = len(inputs)
    sync_contexts = [model for _ in range(length - 1)] + [nullcontext]

    for inp, grad, state, sync_context in zip(inputs, cache, rand_states, sync_contexts):
        with sync_context():
            with state:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    embedding = model(**inp)["last_hidden_state"].mean(axis=1)
            surrogate = torch.dot(embedding.flatten(), grad.flatten())
            surrogate.backward()


def cache_loss(tower1, tower2, query_embeddings, document_embeddings, logit_scale, bidirectional=False):
    # only require grad for embedding / representation
    query_embs = query_embeddings.detach().requires_grad_()
    document_embs = document_embeddings.detach().requires_grad_()

    # I'm not sure this works for LiT
    # TODO: this broke everything with grad cache!
    # no_tower1_sync = getattr(tower1, "no_sync", nullcontext)
    # no_tower2_sync = getattr(tower2, "no_sync", nullcontext)
    no_tower1_sync, no_tower2_sync = nullcontext, nullcontext

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with no_tower1_sync():
            with no_tower2_sync():
                # loss = clip_loss(query_embs, document_embs, logit_scale, gather_enabled=True, bidirectional=bidirectional)
                loss = clip_loss(query_embs, document_embs, logit_scale, bidirectional=bidirectional)
                loss.backward()

    query_cache = query_embs.grad
    document_cache = document_embs.grad

    return query_cache, document_cache, loss.detach()


def grad_cache_loss(tower1, t1_inputs, tower2, t2_inputs, chunk_size, logit_scale, bidirectional=False):
    total_bs = t1_inputs["input_ids"].shape[0]
    chunked_queries = []
    chunked_documents = []

    for chunk_start in range(0, total_bs, chunk_size):
        query_chunk = {k: v[chunk_start : chunk_start + chunk_size] for k, v in t1_inputs.items()}
        chunked_queries.append(query_chunk)

        document_chunk = {k: v[chunk_start : chunk_start + chunk_size] for k, v in t2_inputs.items()}
        chunked_documents.append(document_chunk)

    query_embs, query_rand_states = get_chunked_embeddings(tower1, chunked_queries)
    document_embs, doc_rand_states = get_chunked_embeddings(tower2, chunked_documents)

    query_cache, document_cache, loss = cache_loss(
        tower1, tower2, query_embs, document_embs, logit_scale, bidirectional=bidirectional
    )

    chunked_query_cache = query_cache.split(chunk_size)
    chunked_document_cache = document_cache.split(chunk_size)

    accumulate_gradients(tower1, chunked_queries, chunked_query_cache, query_rand_states)
    # if tower2.training:
    #     accumulate_gradients(tower2, chunked_documents, chunked_document_cache, doc_rand_states)

    return loss


if __name__=='__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/mlm/step_130000")

    # Move model to CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ====== Step 2: Prepare Input Data ======
    queries = [
        "What is artificial intelligence?",
        "How does deep learning work?",
        "What are the applications of machine learning?",
        "Explain the concept of natural language processing.",
        "What is the difference between supervised and unsupervised learning?",
        "How do convolutional neural networks work?",
        "What are transformers in deep learning?",
        "How is reinforcement learning used in robotics?"
    ]

    documents = [
        "Artificial intelligence is the ability of machines to perform tasks that typically require human intelligence, such as reasoning and learning.",
        "Deep learning is a subfield of machine learning that uses neural networks with multiple layers to model complex patterns in data.",
        "Machine learning is used in various fields, including healthcare, finance, marketing, and self-driving cars, to automate decision-making processes.",
        "Natural language processing (NLP) is a branch of AI that enables computers to understand, interpret, and generate human language.",
        "Supervised learning requires labeled data for training, whereas unsupervised learning identifies patterns in data without explicit labels.",
        "Convolutional neural networks (CNNs) process grid-like data, such as images, by using layers of convolutional filters to extract features.",
        "Transformers are a deep learning architecture based on self-attention mechanisms, commonly used in NLP models like BERT and GPT.",
        "Reinforcement learning in robotics allows autonomous agents to learn optimal actions through trial and error to maximize rewards."
    ]


    # Tokenize inputs
    t1_inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(device)
    t2_inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt").to(device)
    # print(t1_inputs)
    # Define logit scale (simulating a trainable parameter)
    logit_scale = lambda x: x.exp()

    # ====== Step 3: Compute Loss ======
    chunk_size = 2  # Small batch size for testing

    loss = grad_cache_loss(
        model, t1_inputs, model, t2_inputs, chunk_size, logit_scale, bidirectional=True
    )

    print(f"Test Loss: {loss.item()}")

    # ====== Step 4: Perform Backpropagation & Optimization ======
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Backpropagation completed successfully.")