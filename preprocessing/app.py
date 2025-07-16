# service_preprocessing/app.py

import pandas as pd
import re
import string
import nltk
from flask import Flask, request, jsonify

# --- Inisialisasi library NLP ---
# Download data yang diperlukan oleh NLTK (hanya perlu sekali)
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Buat instance stemmer dan stopword remover sekali saja saat aplikasi dimulai
# untuk efisiensi.
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()
list_stopwords = StopWordRemoverFactory().get_stop_words()

# Inisialisasi aplikasi Flask
app = Flask(__name__)


# --- Fungsi-fungsi Preprocessing ---

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text); text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r"http\S+", '', text); text = re.sub(r'[0-9]+', '', text)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text); text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation)); text = text.strip(' ')
    return text
def tokenizingText(text): return word_tokenize(text)
def filteringText(text): return [word for word in text if word not in list_stopwords]
def stemmingText(text): return [stemmer.stem(word) for word in text]


# --- Route / Endpoint ---

@app.route('/preprocess', methods=['POST'])
def preprocess_endpoint():
    """
    Endpoint untuk menerima data, melakukan preprocessing, dan mengembalikannya.
    """
    # 1. Ambil data JSON dari request
    request_data = request.get_json()
    if not request_data or 'data' not in request_data:
        return jsonify({"error": "Payload JSON tidak valid. Key 'data' tidak ditemukan."}), 400

    # 2. Konversi list of dictionaries menjadi DataFrame
    df = pd.DataFrame(request_data['data'])

    # Pastikan kolom 'full_text' ada
    if 'full_text' not in df.columns:
        return jsonify({"error": "DataFrame harus memiliki kolom 'full_text'."}), 400

    # 3. Terapkan fungsi preprocessing ke setiap baris di kolom 'full_text'
    #    dan simpan hasilnya di kolom baru 'cleaned_text'.
    df['clean_text'] = df['full_text'].apply(cleaningText)
    df['tokenize_text'] = df['clean_text'].apply(tokenizingText)
    df['filter_text'] = df['tokenize_text'].apply(filteringText)
    df['stem_text'] = df['filter_text'].apply(stemmingText)
    df = df[df['clean_text'].str.strip() != '']
    df.drop_duplicates(subset=['clean_text'], inplace=True)
    
    # 4. Konversi DataFrame yang sudah diperkaya kembali ke format JSON
    #    untuk dikirim sebagai respons.
    response_data = df.to_dict(orient='records')
    
    # 5. Kembalikan data yang sudah bersih.
    #    Struktur payload { "data": [...] } dijaga agar konsisten.
    return jsonify({"data": response_data})


if __name__ == '__main__':
    # Menjalankan server di port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)
