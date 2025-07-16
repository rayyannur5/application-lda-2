# service_main_app/app.py

import os
import io
import pandas as pd
import requests
from flask import Flask, request, jsonify, render_template, session, redirect, url_for

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi URL untuk setiap microservice
# Dalam skenario nyata, ini bisa diambil dari environment variables
SERVICE_URLS = {
    "preprocessing": "http://localhost:5001/preprocess",
    "sentiment": "http://localhost:5002/analyze_sentiment",
    "policy": "http://localhost:5003/classify_policy",
    "lda": "http://localhost:5004/model_lda",
    "summary": "http://localhost:5005/generate_summary"
}

# --- Helper Function untuk memanggil service lain ---
def call_service(service_name, data_payload):
    """
    Fungsi untuk memanggil microservice lain dengan penanganan error.
    """
    try:
        url = SERVICE_URLS.get(service_name)
        if not url:
            return {"error": f"Service '{service_name}' tidak ditemukan."}
        

        if service_name == 'summary':
            response = requests.post(url, json={'data': data_payload['data']['df'], 'prompt': data_payload['data']['lda_prompt']}, timeout=180) # Timeout 180 detik
        else :
            response = requests.post(url, json=data_payload, timeout=180) # Timeout 180 detik

        response.raise_for_status()  # Akan raise error jika status code bukan 2xx
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Gagal terhubung ke service '{service_name}' di {url}."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Terjadi error saat memanggil service '{service_name}': {e}"}


# --- Routes / Endpoints ---

@app.route('/')
def home():
    if 'user' in session: return render_template('home.html', title="Home Page")
    else: return redirect(url_for('login'))

# Rute-rute dan handler lainnya tidak perlu diubah
@app.route('/login', methods=['POST','GET'])
def login():
    if request.method == 'POST':
        username = request.form.get('username'); password = request.form.get('password')
        if username == '1234' and password == '1234':
            session['user'] = username
            return {'message': 'success', 'data': {'nextRoute': '/'}}, 200
        else: return {'message': 'username dan password salah'}, 400

    return render_template('login.html', title="Login Page")

@app.route('/logout')
def logout():
    session.pop('user', None); return redirect(url_for('login'))

@app.route('/process', methods=['POST'])
def process_csv_file():
    """
    Endpoint untuk menerima file CSV dan mengorkestrasi alur kerja analisis
    secara sekuensial dan eksplisit.
    """
    # 1. Validasi input file
    if 'file' not in request.files:
        return jsonify({"error": "File CSV tidak ditemukan. Mohon sertakan dengan key 'file'."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Nama file kosong."}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Format file tidak valid. Hanya menerima .csv"}), 400

    try:
        # 2. Baca dan konversi CSV menjadi format JSON yang bisa dikirim
        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        
        if 'full_text' not in df.columns:
             return jsonify({"error": "File CSV harus memiliki kolom bernama 'full_text'."}), 400

        # Data awal dalam format yang siap dikirim
        initial_data = {"data": df.to_dict(orient='records')}
        
        # --- Alur Kerja Orkestrasi Sekuensial Eksplisit ---

        # Langkah 1: Panggil service Preprocessing
        preprocessed_result = call_service("preprocessing", initial_data)
        if "error" in preprocessed_result:
            status_code = preprocessed_result.get("status_code", 500)
            return jsonify({"error_in_service": "preprocessing", "details": preprocessed_result["error"]}), status_code

        # Langkah 2: Panggil service Sentiment (input dari preprocessing)
        sentiment_result = call_service("sentiment", preprocessed_result)
        if "error" in sentiment_result:
            status_code = sentiment_result.get("status_code", 500)
            return jsonify({"error_in_service": "sentiment", "details": sentiment_result["error"]}), status_code

        # Langkah 3: Panggil service Policy (input dari sentiment)
        policy_result = call_service("policy", sentiment_result)
        if "error" in policy_result:
            status_code = policy_result.get("status_code", 500)
            return jsonify({"error_in_service": "policy", "details": policy_result["error"]}), status_code
        
        # Langkah 4: Panggil service LDA (input dari policy)
        lda_result = call_service("lda", policy_result)
        if "error" in lda_result:
            status_code = lda_result.get("status_code", 500)
            return jsonify({"error_in_service": "lda", "details": lda_result["error"]}), status_code

        # Langkah 5: Panggil service Summary (input dari lda)
        summary_result = call_service("summary", lda_result)
        if "error" in summary_result:
            status_code = summary_result.get("status_code", 500)
            return jsonify({"error_in_service": "summary", "details": summary_result["error"]}), status_code

        # Hasil akhir adalah output dari service terakhir
        return jsonify({
            "status": "success",
            "message": "File berhasil diproses oleh semua service.",
            "final_result": lda_result
        })

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan internal di main_app: {str(e)}"}), 500


if __name__ == '__main__':
    # Menjalankan server di port 5000 dan bisa diakses dari luar
    app.config['SECRET_KEY'] = 'kunciii'

    app.run(host='0.0.0.0', port=5000, debug=True)
