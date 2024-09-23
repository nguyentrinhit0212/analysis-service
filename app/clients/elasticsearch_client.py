from elasticsearch import Elasticsearch
from app.config import Config
from app.utils.logger import AppLogger  # Import the custom logger

class ElasticClient:
    """
    Class to handle interactions with Elasticsearch.
    """

    def __init__(self):
        """
        Initialize the Elasticsearch client using configuration settings.
        """
        self.client = Elasticsearch(Config.ES_HOST)
        self.logger = AppLogger().get_logger()  # Get the logger instance

    def create_index(self, index_name, body):
        """
        Create an Elasticsearch index if it does not exist.
        
        Args:
            index_name (str): The name of the index to create.
            body (dict): The settings and mappings for the index.
        """
        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(index=index_name, body=body)
            self.logger.info(f"Index '{index_name}' created.")
        else:
            self.logger.info(f"Index '{index_name}' already exists.")

    def index_document(self, index_name, doc_id, document):
        """
        Index a single document in the Elasticsearch index.
        
        Args:
            index_name (str): The name of the index.
            doc_id (str): The document ID.
            document (dict): The document to be indexed.
        """
        self.client.index(index=index_name, id=doc_id, document=document)
        self.logger.info(f"Document {doc_id} indexed in '{index_name}'.")

    def index_documents(self, index_name, documents):
        """
        Index multiple documents into the Elasticsearch index in bulk.
        
        Args:
            index_name (str): The name of the Elasticsearch index.
            documents (list): A list of documents to be indexed.
        """
        bulk_body = []
        for doc in documents:
            action = {
                "index": {  # Use 'index' to indicate the action
                    "_index": index_name,
                    "_id": doc.get("id"),
                },
            }
            bulk_body.append(action)
            bulk_body.append(doc)
        if bulk_body:
            response = self.client.bulk(body=bulk_body, refresh="wait_for")
            self.logger.info(f"Indexed {len(documents)} documents into '{index_name}'.")
        else:
            self.logger.info("No documents to index.")

    def search(self, index_name, query):
        """
        Search for documents in the Elasticsearch index based on a query.
        
        Args:
            index_name (str): The name of the index to search.
            query (dict): The query to execute.
        
        Returns:
            list: A list of search results.
        """
        response = self.client.search(index=index_name, body=query)
        self.logger.info(f"Search executed on index '{index_name}' with query: {query}")
        return response

    def delete_index(self, index_name):
        """
        Delete an Elasticsearch index.
        
        Args:
            index_name (str): The name of the index to delete.
        """
        self.client.indices.delete(index=index_name)
        self.logger.info(f"Index '{index_name}' deleted.")
