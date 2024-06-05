from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import torch
app = Flask(__name__)

from transformers import pipeline

classes = ['Appellant',
 'Argument by Appellant',
 'Argument by Defendant',
 'Argument by Petitioner',
 'Argument by Respondent',
 'Conclusion',
 'Court Discourse',
 'Fact',
 'Issue',
 'Judge',
 'Petitioner',
 'Precedent Analysis',
 'Ratio',
 'Respondent',
 'Section Analysis']

tokenizer = AutoTokenizer.from_pretrained("xshubhamx/InLegalBERT")
bert_model = AutoModelForSequenceClassification.from_pretrained("xshubhamx/InLegalBERT")

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower() 
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.form['paragraph']
    paragraph = data.strip()
    paragraph_arr = [str(p) for p in paragraph.split('\n') if p.strip()]
    print(paragraph_arr)

    pipe = pipeline("text-classification", model="xshubhamx/InLegalBERT")
    print(pipe(paragraph_arr))
    res = pipe(paragraph_arr)
    print(res)
    class_names = []
    for i in res:
        class_names.append(i['label'])
    response = {'paragraph': paragraph_arr, 'classes': class_names}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
