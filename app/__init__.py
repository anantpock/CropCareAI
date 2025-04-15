import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

db = SQLAlchemy()

def create_app():
    # Build absolute paths to the templates and static folders.
    base_dir = os.path.dirname(os.path.realpath(__file__))
    template_path = os.path.join(base_dir, '..', 'templates')
    static_path = os.path.join(base_dir, '..', 'static')
    
    app = Flask(__name__, template_folder=template_path, static_folder=static_path)
    
    app.secret_key = os.environ.get("SESSION_SECRET", "dev")
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", "sqlite:///plant_disease.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    CORS(app)

    # Import routes and register blueprint after app creation
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    with app.app_context():
        db.create_all()

    return app
