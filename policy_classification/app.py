import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
from transformers import pipeline, AutoTokenizer
from transformers import logging as hf_logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify

hf_logging.set_verbosity_error()

app = Flask(__name__)

MODEL_PATH = os.path.abspath("./policy_classification/model") 

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Direktori model tidak ditemukan di: {MODEL_PATH}")


tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
policy_model = pipeline(
    "sentiment-analysis", 
    model=MODEL_PATH,
    tokenizer=tokenizer
)

def classify_policy(text):
    res = policy_model(str(text))
    label_map = {'LABEL_0': 'Jatim Agro', 'LABEL_1': 'Jatim Akses', 'LABEL_2': 'Jatim Amanah', 'LABEL_3': 'Jatim Berdaya', 'LABEL_4': 'Jatim Berkah', 'LABEL_5': 'Jatim Cerdas dan Sehat', 'LABEL_6': 'Jatim Harmoni', 'LABEL_7': 'Jatim Kerja', 'LABEL_8': 'Jatim Sejahtera'}
    return label_map.get(res[0]['label'], 'Lainnya')

# --- Route / Endpoint ---

@app.route('/classify_policy', methods=['POST'])
def policy_endpoint():

    request_data = request.get_json()
    if not request_data or 'data' not in request_data:
        return jsonify({"error": "Payload JSON tidak valid. Key 'data' tidak ditemukan."}), 400


    df = pd.DataFrame(request_data['data'])


    if 'clean_text' not in df.columns:
        return jsonify({"error": "DataFrame harus memiliki kolom 'clean_text'."}), 400

    df['policy'] = df['clean_text'].apply(classify_policy)

    response_data = df.to_dict(orient='records')

    return jsonify({"data": response_data})

if __name__ == '__main__':
    app.run(port=5003, debug=True)
