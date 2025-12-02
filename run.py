#!/usr/bin/env python3
"""
Flash Detector - Main Entry Point

Run this script to start the web application.

Usage:
    python run.py [--host HOST] [--port PORT] [--debug]

Example:
    python run.py --host 0.0.0.0 --port 5000 --debug
"""

import argparse
import os
import logging
from app import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description='Flash Detector Web Application')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    app = create_app()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                   FLASH DETECTOR                         ║
║              Industrial Vision Inspection                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║   Server starting at: http://{args.host}:{args.port}                ║
║                                                          ║
║   Open your browser and navigate to the URL above        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
