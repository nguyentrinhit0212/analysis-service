from elasticsearch import Elasticsearch
from app.config import ES_HOST

es_client = Elasticsearch(ES_HOST)

def create_index(index_name, body):
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=body)
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")

def index_document(index_name, doc_id, document):
    es_client.index(index=index_name, id=doc_id, document=document)
    print(f"Document {doc_id} indexed in '{index_name}'.")

def search_documents(index_name, query):
    response = es_client.search(index=index_name, body=query)
    return response['hits']['hits']

def delete_index(index_name):
    es_client.indices.delete(index=index_name)
    print(f"Index '{index_name}' deleted.")