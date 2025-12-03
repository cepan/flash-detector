"""
Flash Detector Web Application

A web-based tool for detecting flash defects in industrial parts
by comparing test images against a golden reference sample.
"""

import os
from flask import Flask, send_from_directory, make_response
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

    # Auto-load saved reference on startup (if exists)
    with app.app_context():
        try:
            from .api.routes import load_saved_reference_on_startup
            load_saved_reference_on_startup()
        except Exception as e:
            print(f"Could not auto-load saved reference: {e}")

    # Serve index.html at root (with no-cache headers)
    @app.route('/')
    def index():
        response = make_response(send_from_directory(app.static_folder, 'index.html'))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    # Serve other static files
    @app.route('/<path:path>')
    def static_files(path):
        return send_from_directory(app.static_folder, path)
    
    return app


# Create app instance for direct running
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
