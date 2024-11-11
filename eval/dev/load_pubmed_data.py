import json
import re
from Bio import Entrez  # Install with 'pip install biopython'
from Bio.Entrez import efetch, read

# Register your email
Entrez.email = "your@email.com"

def clean_html(raw_html):
    # Use regex to remove HTML tags
    clean_text = re.sub('<.*?>', '', raw_html)
    return clean_text

def get_article_details(pmid):
    # Call PubMed API to fetch article details
    handle = efetch(db='pubmed', id=str(pmid), retmode='xml')
    xml_data = read(handle)

    # Access the PubMedArticle data from the XML response
    article_data = xml_data['PubmedArticle'][0]

    # Extract title (use None if not found)
    title = article_data['MedlineCitation']['Article'].get('ArticleTitle', '')

    # Extract abstract and clean HTML tags (use None if not found)
    abstract_data = article_data['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', '')
    abstract = clean_html(abstract_data[0]) if abstract_data else ''

    # Extract MeSH terms as a list (use empty list if not found)
    mesh_terms = []
    if 'MeshHeadingList' in article_data['MedlineCitation']:
        mesh_terms = [mesh['DescriptorName'].title() for mesh in article_data['MedlineCitation']['MeshHeadingList']]

    # Extract keywords as a list (use empty list if not found)
    keywords = []
    if 'KeywordList' in article_data['MedlineCitation']:
        for keyword_list in article_data['MedlineCitation']['KeywordList']:
            keywords.extend([str(keyword) for keyword in keyword_list])

    return {
        "title": title,
        "abstract": abstract,
        "mesh_terms": mesh_terms,
        "keywords": keywords
    }

def fetch_and_save_articles(query, max_articles=1000, output_file="medical_data.json"):
    # Search PubMed for articles related to the query
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_articles)
    record = Entrez.read(handle)
    handle.close()

    # Get the list of PubMed IDs
    id_list = record["IdList"]

    articles = []

    # Fetch details for each article and append to the list
    for pmid in id_list:
        try:
            article_details = get_article_details(pmid)
            articles.append(article_details)
        except Exception as e:
            print(f"Error fetching data for PMID {pmid}: {e}")

    # Save the articles to a JSON file
    with open(output_file, mode='w', encoding='utf-8') as file:
        json.dump(articles, file, indent=4)

    print(f"Data saved to {output_file}")

# Example usage with the comprehensive query
comprehensive_query = (
    "(treatment OR therapy OR intervention) AND "
    "(risk assessment OR risk management OR evaluation) AND "
    "(hospital OR intensive care OR emergency room OR clinical care OR healthcare)"
)
fetch_and_save_articles(query=comprehensive_query, max_articles=500)
