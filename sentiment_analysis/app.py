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

MODEL_PATH = os.path.abspath("./sentiment_analysis/model") 

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Direktori model tidak ditemukan di: {MODEL_PATH}")

sentiment_model = pipeline(
    "sentiment-analysis", 
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

def classify_sentiment(text):
    res = sentiment_model(str(text))
    return 'positive' if res[0]['label'] == 'neutral' else res[0]['label']


# --- Route / Endpoint ---

@app.route('/analyze_sentiment', methods=['POST'])
def sentiment_endpoint():

    request_data = request.get_json()
    if not request_data or 'data' not in request_data:
        return jsonify({"error": "Payload JSON tidak valid. Key 'data' tidak ditemukan."}), 400


    df = pd.DataFrame(request_data['data'])


    if 'clean_text' not in df.columns:
        return jsonify({"error": "DataFrame harus memiliki kolom 'clean_text'."}), 400


    df['sentiment'] = df['clean_text'].apply(classify_sentiment)
    plt.figure(figsize=(6, 4)); sns.countplot(x=df["sentiment"], palette=['#ff4f4f', '#0cad00', '#C0C0C0']); plt.savefig('storage/plot_sentiment.png'); plt.close()
    plt.figure(figsize=(4, 4)); counts = df["sentiment"].value_counts(); colors = {'negative': '#ff4f4f', 'positive': '#0cad00', 'neutral': '#C0C0C0'}; plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=[colors.get(label, '#C0C0C0') for label in counts.index], startangle=140); plt.savefig('storage/pie_sentiment.png'); plt.close()
    

    response_data = df.to_dict(orient='records')
    

    return jsonify({"data": response_data})

if __name__ == '__main__':
    # Menjalankan server di port 5002
    app.run(host='0.0.0.0', port=5002, debug=True)
