from flask import Blueprint, jsonify, request
from app.utils.logger import AppLogger  # Import logger
from app.models.bert_model import BertModel  # Import the classifier
from app.clients.elasticsearch_client import ElasticClient  # Import the Elasticsearch client

class PostAnalyzer:
    def __init__(self):
        """
        Initialize the PostAnalyzer class and create a Flask Blueprint.
        """
        self.logger = AppLogger().get_logger()
        self.main = Blueprint('main', __name__)
        self.bert_model = BertModel('yahoo_answers/checkpoint-78750')  # Initialize the BERT model
        self.elastic_client = ElasticClient()  # Initialize the Elasticsearch client
        # Add routes to the blueprint
        self.main.add_url_rule('/analyze_post', 'analyze_post', self.analyze_post, methods=['POST'])
        self.main.add_url_rule('/topic_trends', 'topic_trends', self.get_topic_trends, methods=['GET'])

    def analyze_post(self):
        """
        API endpoint to analyze a post using BERT Topic Classification.
        """
        data = request.get_json()
        post_content = data.get('content', '')

        if not post_content:
            return jsonify({'error': 'No content provided'}), 400

        # Classify topic using BERT
        topic = self.bert_model.classify_topic(post_content)

        # Log the topic
        self.logger.info(f"Predicted topic: {topic}")

        # Return analysis result
        result = {
            'content': post_content,
            'topic': topic,
        }

        return jsonify(result), 200

    def get_topic_trends(self):
        """
        API endpoint to count the occurrences of topics in Elasticsearch.
        """
        try:
            # Query Elasticsearch to get topic counts
            es_query = {
                "size": 0,  # No document hits, only aggregations
                "aggs": {
                    "topics_count": {
                        "nested": {
                            "path": "topics"  # Define the nested field path
                        },
                        "aggs": {
                            "topic_name_count": {
                                "terms": {
                                    "field": "topics.name",  # Aggregate by the topic name
                                    "size": 10  # Adjust as necessary
                                }
                            }
                        }
                    }
                }
            }

            # Execute the query
            es_response = self.elastic_client.search(index_name='reddit_posts', query=es_query)
            # Process the aggregation results
            buckets = es_response['aggregations']['topics_count']['topic_name_count']['buckets']
        
            # Build the result as a list of topic name and count pairs
            topic_counts = [[bucket['key'], bucket['doc_count']] for bucket in buckets]

            # Return the result as a JSON response
            return jsonify(topic_counts), 200

        except Exception as e:
            self.logger.error(f"Error in get_topic_trends: {e}")
            return jsonify({'error': str(e)}), 500

    def get_blueprint(self):
        """
        Return the Flask Blueprint with registered routes.
        """
        return self.main
