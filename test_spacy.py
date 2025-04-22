import spacy

try:
    nlp = spacy.load('en_core_web_sm')
    print('Model loaded successfully!')
except Exception as e:
    print(f'Error loading model: {str(e)}') 