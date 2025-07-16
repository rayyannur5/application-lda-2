# service_main_app/app.py

import eventlet
eventlet.monkey_patch()

import os
import io
import pandas as pd
import numpy as np
import requests
import uuid
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room

# --- Inisialisasi Aplikasi ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunciii_rahasia_banget'
# Buat folder untuk menyimpan file upload sementara
os.makedirs('uploads', exist_ok=True) 
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SERVER_NAME'] = 'localhost:5000'

# Inisialisasi SocketIO
socketio = SocketIO(app)

# --- Konfigurasi Layanan (Sama seperti sebelumnya) ---
SERVICE_URLS = {
    "preprocessing": "http://localhost:5001/preprocess",
    "sentiment": "http://localhost:5002/analyze_sentiment",
    "policy": "http://localhost:5003/classify_policy",
    "lda": "http://localhost:5004/model_lda",
    "summary": "http://localhost:5005/generate_summary"
}

# --- Fungsi Helper (Sama seperti sebelumnya) ---
def call_service(service_name, data_payload):
    # try:
    url = SERVICE_URLS.get(service_name)
    if not url:
        return {"error": f"Service '{service_name}' tidak ditemukan."}
    
    # Logika custom untuk service summary
    if service_name == 'summary':
        response = requests.post(url, json={'data': data_payload['data']['df'], 'prompt': data_payload['data']['lda_prompt']}, timeout=180)
    else:
        response = requests.post(url, json=data_payload, timeout=180)
    
    response.raise_for_status()
    return response.json()
    # except requests.exceptions.RequestException as e:
    #     # Mengembalikan error dalam format yang bisa di-handle
    #     return {"error": f"Error di service '{service_name}': {str(e)}"}

# --- FUNGSI PROSES LATAR BELAKANG (Sama seperti sebelumnya) ---
def run_processing_pipeline(file_path, user_room):
    """
    Fungsi ini menjalankan seluruh pipeline dan mengirim update ke 'room' pengguna.
    `user_room` adalah ID unik pengguna dari session.
    """
    current_data = None
    
    try:
        # Pastikan kita punya user_room untuk mengirim update
        if not user_room:
            raise Exception("Tidak dapat memulai proses, sesi pengguna tidak ditemukan.")
        
        socketio.sleep(1)

        # Baca file CSV yang sudah diunggah
        df = pd.read_csv(file_path)
        if 'full_text' not in df.columns:
            raise ValueError("File CSV harus memiliki kolom bernama 'full_text'.")
        
        df = df.replace([np.inf, -np.inf], np.nan)  # ubah inf jadi NaN
        df = df.fillna("") 
        
        current_data = {"data": df.to_dict(orient='records')}

        # --- Alur Kerja Orkestrasi Sekuensial Eksplisit ---

        # Langkah 1: Preprocessing
        socketio.emit('step_start', {'step': 'preprocessing'})
        preprocessed_result = call_service("preprocessing", current_data)
        if "error" in preprocessed_result:
            raise Exception(preprocessed_result["error"])
        socketio.emit('step_end', {'step': 'preprocessing', 'message': "Langkah 'preprocessing' berhasil diselesaikan."}, to=user_room)

        # Langkah 2: Sentiment Analysis
        socketio.emit('step_start', {'step': 'sentiment'}, to=user_room)
        sentiment_result = call_service("sentiment", preprocessed_result)
        if "error" in sentiment_result:
            raise Exception(sentiment_result["error"])
        socketio.emit('step_end', {'step': 'sentiment', 'message': "Langkah 'sentiment' berhasil diselesaikan."}, to=user_room)

        # Langkah 3: Policy Classification
        socketio.emit('step_start', {'step': 'policy'}, to=user_room)
        policy_result = call_service("policy", sentiment_result)
        if "error" in policy_result:
            raise Exception(policy_result["error"])
        socketio.emit('step_end', {'step': 'policy', 'message': "Langkah 'policy' berhasil diselesaikan."}, to=user_room)

        # Langkah 4: LDA Modeling
        socketio.emit('step_start', {'step': 'lda'}, to=user_room)
        lda_result = call_service("lda", policy_result)
        if "error" in lda_result:
            raise Exception(lda_result["error"])
        current_data = lda_result
        socketio.emit('step_end', {'step': 'lda', 'message': "Langkah 'lda' berhasil diselesaikan."}, to=user_room)

        # Langkah 5: Summary
        socketio.emit('step_start', {'step': 'summary'}, to=user_room)
        summary_result = call_service("summary", lda_result)
        if "error" in summary_result:
            raise Exception(summary_result["error"])
        socketio.emit('step_end', {'step': 'summary', 'message': "Langkah 'summary' berhasil diselesaikan."}, to=user_room)
        
        # --- Proses Selesai ---
        # Simpan hasil akhir ke file CSV di folder static agar bisa di-download
        final_df = pd.DataFrame(lda_result['data']['df'])
        final_csv_path = os.path.join('static', 'result.csv')
        final_df.to_csv(final_csv_path, index=False)

        sentiment = final_df["sentiment"].value_counts().to_dict()
        kebijakan = final_df["policy"].value_counts().to_dict()

        df_hasil_head = final_df.groupby(['sentiment', 'topic']).head(5).reset_index(drop=True).drop(['clean_text', 'tokenize_text', 'filter_text', 'stem_text'], axis=1).to_json()

        with app.app_context():
            result_html = render_template(
                'result.html', 
                lda=lda_result['data']['html'], 
                sentiment=sentiment, 
                kebijakan=kebijakan
            )
        
        # Kirim event 'process_complete' dengan hasil akhir
        socketio.emit('process_complete', {
            "message": "Semua proses berhasil diselesaikan.",
            "html": result_html,
            "summary": summary_result['data'],
            "df_hasil_head": df_hasil_head,
        }, to=user_room)

    except Exception as e:
        # Jika terjadi error di mana pun dalam pipeline
        socketio.emit('process_error', {'error': str(e)}, to=user_room)
        print(e)
    finally:
        # Hapus file sementara setelah selesai
        if os.path.exists(file_path):
            os.remove(file_path)

# --- Rute HTTP (Sama seperti sebelumnya) ---

@app.route('/process', methods=['POST'])
def process_csv_file():
    if 'user' not in session:
        return jsonify({"error": "Sesi tidak valid. Silakan login kembali."}), 401

    if 'file' not in request.files:
        return jsonify({"error": "File CSV tidak ditemukan."}), 400
    
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({"error": "Mohon pilih file CSV yang valid."}), 400

    filename = str(uuid.uuid4()) + ".csv"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    user_room_id = session['user']
    socketio.start_background_task(run_processing_pipeline, file_path, user_room_id)
    
    return jsonify({"message": "Proses dimulai. Tunggu pembaruan log."}), 202

# --- Rute Lainnya (Sama seperti sebelumnya) ---
@app.route('/')
def home():
    if 'user' in session: return render_template('home.html', title="Home Page")
    else: return redirect(url_for('login'))

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


# --- Event Handler WebSocket (Sama seperti sebelumnya) ---
@socketio.on('connect')
def handle_connect():
    # Hanya log koneksi, tidak lagi otomatis join room di sini
    print(f"Klien terhubung dengan SID: {request.sid}")

    if 'user' in session:
        user_room_id = session['user']
        join_room(user_room_id)
        print(f"Klien dengan SID {request.sid} secara eksplisit dimasukkan ke room: '{user_room_id}'")
    else:
        print(f"Klien anonim dengan SID {request.sid} mencoba join, tapi tidak ada sesi.")


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Klien terputus dengan SID: {request.sid}")

# --- Menjalankan Aplikasi (DIUBAH) ---
if __name__ == '__main__':
    # Gunakan socketio.run() dan biarkan ia menggunakan eventlet secara otomatis.
    # Menghapus 'allow_unsafe_werkzeug=True' karena tidak lagi diperlukan.
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
