# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 22:25:54 2024

@author: user
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
