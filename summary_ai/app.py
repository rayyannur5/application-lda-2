import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify

from google import genai
from google.genai import types

app = Flask(__name__)


def generate_with_gemini(prompt):
    """
    Mengirimkan prompt ke Gemini API dan mengembalikan respons sebagai teks lengkap.
    """
    try:
        client = genai.Client(api_key='AIzaSyA9dVec3gJ7tCsbyakFi4vOxtUDLzBgUfQ')

        model = "gemini-2.5-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config = types.ThinkingConfig(
                thinking_budget=-1,
            ),
            response_mime_type="text/plain",
        )

        # response = client.models.generate_content(
        #     model=model,
        #     contents=contents,
        #     config=generate_content_config,
        # )

        responses = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")
            responses += chunk.text

        return responses
    except Exception as e:
        error_message = f"Gagal menghasilkan teks dengan Gemini: {e}"
        print(error_message) # Juga print ke konsol untuk debugging
        return f"Terjadi kesalahan saat berkomunikasi dengan API Gemini. Pastikan API Key Anda valid dan coba lagi. \n\nDetail Error: {e}"



# --- Route / Endpoint ---

@app.route('/generate_summary', methods=['POST'])
def summary_endpoint():

    request_data = request.get_json()
    if not request_data or 'data' not in request_data:
        return jsonify({"error": "Payload JSON tidak valid. Key 'data' tidak ditemukan."}), 400


    df = pd.DataFrame(request_data['data'])


    if 'clean_text' not in df.columns:
        return jsonify({"error": "DataFrame harus memiliki kolom 'clean_text'."}), 400
    

    df_hasil_head = df.groupby(['sentiment', 'topic']).head(5).reset_index(drop=True).drop(['clean_text', 'tokenize_text', 'filter_text', 'stem_text'], axis=1).to_json()


    prompt = f"""
Analisis Data Aspirasi Masyarakat Terkait Kebijakan Pemerintah Provinsi.
**DATA YANG DISEDIAKAN:**

1.  **Pemodelan Topik (LDA):** Tiga topik utama yang muncul dari data adalah:
    {request_data['prompt']}

2.  **Sebaran Sentimen:**
    {str(df["sentiment"].value_counts())}

3.  **Sebaran Topik:**
    {str(df["topic"].value_counts())}

4.  **Sampel Data:** Berikut adalah beberapa contoh data mentah yang telah diklasifikasikan berdasarkan sentimen dan topik.
    ```json
    {df_hasil_head}
    ```
5. **Sebaran Kebijakan**
    {str(df["policy"].value_counts())}

6. **Data Kebijakan** berikut data kota terdampak pada setiap kebijakan
    - Jatim Agro : Batu, Madiun, Kota Madiun, Nganjuk, Pasuruan, Kediri, Magetan, Probolinggo, Sampang, Bondowoso
    - Jatim Akses : Banyuwangi, Trenggalek, Malang, Bondowoso, Tulungagung, Ponorogo, Madiun, Nganjuk, Kediri, Sumenep
    - Jatim Amanah : Surabaya, Malang, Sidoarjo, Sampang, Probolinggo
    - Jatim Berdaya : Surabaya, Malang, Madiun, Trenggalek, Mojokerto, Sidoarjo
    - Jatim Berkah : Sampang, Sumenep, Sidoarjo, Jombang, Banyuwangi, Situbondo, Tuban
    - Jatim Cerdas dan Sehat :
        - Masalah Pendidikan : Surabaya, Malang, Sidoarjo, Bangkalan, Bojonegoro, Probolinggo, Blitar
        - Masalah Kesehatan : Malang, Mojokerto, Batu, Madiun, Jember, Banyuwangi, Madura, Probolinggo
    - Jatim Harmoni : Jember, Banyuwangi, Ponorogo
    - Jatim Kerja : Sampang, Lumajang, Sumenep, Lamongan, Malang, Madiun, Surabaya, Ponorogo, Pacitan, Pasuruan

**TUGAS ANDA:**

Berdasarkan semua data di atas, berikan analisis komprehensif dengan format berikut:

1.  **Deskripsi Setiap Topik:** Jelaskan secara rinci makna dari setiap topik berdasarkan kata-kata kunci yang ada. Berikan nama yang deskriptif untuk setiap topik.

2.  **Analisis Permasalahan Utama:** Identifikasi dan jelaskan masalah inti atau isu utama yang dihadapi masyarakat berdasarkan korelasi antara topik dan sentimen.

3.  **Analisis Data Sebaran:** Berikan interpretasi terhadap data sebaran sentimen dan topik. Apa yang dapat disimpulkan dari dominasi topik tertentu?

4.  **Konteks Lokal :** Kaitkan dengan data daerah sesuai dengan kebijakan yang paling sering muncul pada sebaran kebijakan untuk mendukung apa yang anda deskripsikan ?

Gunakan bahasa yang profesional, jelas, dan lugas.
"""

    generated_analysis = generate_with_gemini(prompt)

    return jsonify({"data": generated_analysis})

if __name__ == '__main__':
    app.run(port=5005, debug=True)
