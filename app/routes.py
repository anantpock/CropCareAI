from flask import Blueprint, render_template, request, jsonify, session
import os
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid

from app import db
from app.models import User, PlantDiseaseResult

from app.plant_disease_detector import detect_disease
from app.gemini_service import get_treatment_recommendation, chat_with_gemini

main = Blueprint("main", __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@main.before_app_request

def setup():
    upload_folder = os.environ.get("UPLOAD_FOLDER", "static/uploads")
    os.makedirs(upload_folder, exist_ok=True)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/history')
def history():
    results = PlantDiseaseResult.query.order_by(PlantDiseaseResult.timestamp.desc()).limit(20).all()
    return render_template('history.html', results=results)

@main.route('/chat')
def chat():
    disease_id = request.args.get('disease_id')
    disease_result = PlantDiseaseResult.query.filter_by(id=disease_id).first() if disease_id else None
    return render_template('chat.html', disease_result=disease_result)

@main.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_extension = os.path.splitext(filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    upload_folder = os.environ.get("UPLOAD_FOLDER", "static/uploads")
    file_path = os.path.join(upload_folder, unique_filename)

    file.save(file_path)

    try:
        prediction, confidence = detect_disease(file_path)
        result = PlantDiseaseResult(
            image_path=file_path,
            prediction=prediction,
            confidence=float(confidence),
            timestamp=datetime.now()
        )
        db.session.add(result)
        db.session.commit()

        return jsonify({
            'id': result.id,
            'image_path': file_path,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': result.timestamp.isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main.route('/api/get_treatment', methods=['POST'])
def get_treatment():
    data = request.json
    if not data or 'disease' not in data:
        return jsonify({'error': 'Disease name is required'}), 400

    try:
        treatment = get_treatment_recommendation(data['disease'])
        return jsonify({'treatment': treatment}), 200
    except Exception as e:
        logger.error(f"Error getting treatment: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main.route('/api/results', methods=['GET'])
def get_results():
    results = PlantDiseaseResult.query.order_by(PlantDiseaseResult.timestamp.desc()).limit(20).all()
    return jsonify([
        {
            'id': r.id,
            'image_path': r.image_path,
            'prediction': r.prediction,
            'confidence': r.confidence,
            'timestamp': r.timestamp.isoformat()
        } for r in results
    ]), 200

@main.route('/api/result/<int:result_id>', methods=['GET'])
def get_result(result_id):
    result = PlantDiseaseResult.query.get_or_404(result_id)
    return jsonify({
        'id': result.id,
        'image_path': result.image_path,
        'prediction': result.prediction,
        'confidence': result.confidence,
        'timestamp': result.timestamp.isoformat()
    }), 200

@main.route('/api/chat', methods=['POST'])
def handle_chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400

    user_message = data['message']
    session_id = data.get('session_id') or session.get('session_id') or str(uuid.uuid4())
    session['session_id'] = session_id

    try:
        response = chat_with_gemini(session_id, user_message)
        return jsonify({'response': response, 'session_id': session_id}), 200
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@main.app_errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@main.app_errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500
