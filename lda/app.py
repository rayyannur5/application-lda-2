import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
import gensim
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim

app = Flask(__name__)


def lda(df):
    
    id2word = gensim.corpora.Dictionary(df["stem_text"]); corpus = [id2word.doc2bow(text) for text in df["stem_text"]]
    lda_model = LdaModel(corpus, num_topics=3, id2word=id2word, passes=10, random_state=42)

    lda_prompt = ""
    for idx, topic in lda_model.print_topics(-1):
        lda_prompt += '\nTopic: {} \nWords: {}'.format(idx, topic)

    
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False); html = pyLDAvis.prepared_data_to_html(vis)
    def get_max_topics(topics): return max(topics, key=lambda item: item[1])[0] if topics else -1
    df['topic'] = [get_max_topics(lda_model.get_document_topics(item)) for item in corpus]
    
    return html, df, lda_prompt

# --- Route / Endpoint ---

@app.route('/model_lda', methods=['POST'])
def policy_endpoint():

    request_data = request.get_json()
    if not request_data or 'data' not in request_data:
        return jsonify({"error": "Payload JSON tidak valid. Key 'data' tidak ditemukan."}), 400


    df = pd.DataFrame(request_data['data'])


    if 'stem_text' not in df.columns:
        return jsonify({"error": "DataFrame harus memiliki kolom 'stem_text'."}), 400

    
    html, df, lda_prompt = lda(df)

    df_json = df.to_dict(orient='records')

    return jsonify({"data": {"html" : html, "df" : df_json, "lda_prompt": lda_prompt}})

if __name__ == '__main__':
    app.run(port=5004, debug=True)
