from flask import Blueprint, jsonify, request
from app.utils import logger  # Import logger
from app.models import classify_intent  # Import from models package

main = Blueprint('main', __name__)

@main.route('/analyze_post', methods=['POST'])
def analyze_post():
    """
    API endpoint to analyze a post using BERT Intent Classification.
    """
    data = request.get_json()
    post_content = data.get('content', '')

    if not post_content:
        return jsonify({'error': 'No content provided'}), 400

    # Classify intent using BERT
    intent = classify_intent(post_content)

    # Log the intent
    logger.info(f"Predicted intent: {intent}")

    # Return analysis result
    result = {
        'content': post_content,
        'intent': intent,
    }

    return jsonify(result), 200