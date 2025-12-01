"""
Flash Detector Web Application

A web-based tool for detecting flash defects in industrial parts
by comparing test images against a golden reference sample.
"""

import os
from flask import Flask, send_from_directory
from flask_cors import CORS

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                static_folder='static',
                static_url_path='/static')
    
    # Enable CORS for API endpoints
    CORS(app)
    
    # Default configuration
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'uploads')
    app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    
    # Apply custom config
    if config:
        app.config.update(config)
    
    # Ensure folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    # Register API blueprint
    from .api import api
    app.register_blueprint(api, url_prefix='/api')
    
    # Serve index.html at root
    @app.route('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')
    
    # Serve other static files
    @app.route('/<path:path>')
    def static_files(path):
        return send_from_directory(app.static_folder, path)
    
    return app


# Create app instance for direct running
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
