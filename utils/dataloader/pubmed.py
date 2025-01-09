import os
import re
import tqdm
import pandas as pd
from Bio import Entrez
import os
import time
def pubmed_get_article_details(pmid):

    handle = Entrez.efetch(db='pubmed', id=str(pmid), retmode='xml')
    xml_data = Entrez.read(handle)
    handle.close()

    article_data = xml_data['PubmedArticle'][0]
    title = article_data['MedlineCitation']['Article'].get('ArticleTitle', '')
    abstract_data = article_data['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', '')
    abstract = ''.join(abstract_data) if abstract_data else ''

    return {
        "Title": title,
        "Abstract": abstract,
    }

def fetch_pubmed_abstracts(query, email, max_results=100, year=2024):

    Entrez.email = email

    # Search PubMed for the query
    query_with_year = f"{query} AND {year}[DP]"
    handle = Entrez.esearch(db="pubmed", term=query_with_year, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()

    # Fetch article details using IDs
    id_list = record["IdList"]
    if not id_list:
        print("No articles found for the query.")
        return []

    print(f"Found {len(id_list)} articles. Fetching abstracts...")

    abstracts = []
    for pmid in id_list:
        try:
            article_details = pubmed_get_article_details(pmid)
            abstracts.append(article_details)
        except Exception as e:
            print(f"Error fetching data for PMID {pmid}: {e}")

    return abstracts


def save_abstracts_for_mlm(abstracts, output_dir="data",name="no_name"):
    """
    Save abstracts stacked together for MLM training.

    :param abstracts: List of abstracts.
    :param output_dir: Directory to save the abstracts.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"pubmed_mlm_{name}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for abstract in abstracts:
            if abstract["Abstract"] != "N/A":
                f.write(abstract["Abstract"] + "\n")

    print(f"MLM corpus saved to {output_file}")

def save_abstracts_for_contrastive(abstracts, output_dir="data",name="no_name"):
    """
    Save title and abstracts for contrastive learning.

    :param abstracts: List of abstracts.
    :param output_dir: Directory to save the contrastive learning data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"pubmed_cl_{name}.csv")
    df = pd.DataFrame(abstracts)
    df = df[df["Title"].notna() & df["Abstract"].notna()]  # Filter out rows with missing data
    df.to_csv(output_file, index=False, columns=["sentence1", "sentence2"], encoding="utf-8")

    print(f"Contrastive learning data saved to {output_file}")

if __name__ == "__main__":
    # Example query
    query = "Cancer"
    email = "your_email@example.com"  # Replace with your email
    max_results = 2
    year = 2024

    pubmed_abstracts = fetch_pubmed_abstracts(query, email, max_results, year)
    save_abstracts_for_mlm(pubmed_abstracts,name=query.replace(' ','_'))
    save_abstracts_for_contrastive(pubmed_abstracts,name=query.replace(' ','_'))
