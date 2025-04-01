#!/bin/bash

pip install --upgrade pip
pip install nltk sentence-transformers chromadb openai python-dotenv streamlit
python -c "import nltk; nltk.download('gutenberg')" 