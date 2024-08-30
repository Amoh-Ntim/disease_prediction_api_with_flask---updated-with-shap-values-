# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 19:24:21 2024

@author: user
"""

from flask import Flask
from flask_cors import CORS
from .routes import api
import logging
from logging import FileHandler, Formatter
def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.register_blueprint(api, url_prefix='/api')


    # Set up logging
    if not app.debug:
        file_handler = FileHandler('app.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(Formatter('%(asctime)s %(levelname)s: %(message)s'))
        app.logger.addHandler(file_handler)

    return app
