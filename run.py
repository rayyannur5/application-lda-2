# run_all.py

import subprocess
import sys
import time
import os
import threading

# --- KODE WARNA ANSI ---
COLORS = [
    "\033[92m",  # HIJAU
    "\033[96m",  # CYAN
    "\033[93m",  # KUNING
    "\033[94m",  # BIRU
    "\033[95m",  # MAGENTA
    "\033[91m",  # MERAH
]
RESET_COLOR = "\033[0m"

# Daftar semua layanan yang akan dijalankan.
# Key adalah nama deskriptif, Value adalah nama folder.
services = {
    "Main App": "main",
    "Preprocessing": "preprocessing",
    "Sentiment": "sentiment_analysis",
    "Policy": "policy_classification",
    "LDA": "lda",
    "Summary": "summary_ai"
}

processes = []

# --- FUNGSI UNTUK MENAMPILKAN LOG (DENGAN WARNA) ---
def stream_output(pipe, service_name, color):
    """Fungsi ini membaca output dari sebuah proses dan menampilkannya dengan prefix berwarna."""
    try:
        # iter(pipe.readline, '') akan membaca baris demi baris sampai proses selesai.
        for line in iter(pipe.readline, ''):
            # Mencetak dengan format: WARNA[NAMA_SERVICE] PESAN RESET_WARNA
            print(f"{color}[{service_name}]{RESET_COLOR} {line.strip()}", flush=True)
    finally:
        pipe.close()

print("--- Memulai semua layanan microservice ---")

try:
    # Menggunakan sys.executable untuk memastikan menggunakan interpreter python yang sama
    python_executable = sys.executable
    color_index = 0

    for name, folder in services.items():
        # Pastikan path ke folder service benar
        service_path = os.path.join(os.path.dirname(__file__), folder)
        if not os.path.isdir(service_path) or not os.path.exists(os.path.join(service_path, "app.py")):
            print(f"!!! Peringatan: Folder atau app.py untuk '{name}' tidak ditemukan di '{service_path}'. Dilewati.")
            continue
            
        # Perintah untuk menjalankan app.py di dalam folder service
        command = [python_executable, "-u", "app.py"] # Opsi -u untuk unbuffered output
        
        # --- PERUBAHAN UTAMA DI SINI ---
        # Menjalankan perintah sebagai proses baru dan menangkap outputnya.
        process = subprocess.Popen(
            command, 
            cwd=service_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True
        )
        processes.append(process)

        # --- MEMULAI THREAD UNTUK SETIAP LOG (DENGAN WARNA) ---
        # Pilih warna untuk layanan ini
        service_color = COLORS[color_index % len(COLORS)]
        color_index += 1
        
        # Membuat dan memulai thread terpisah untuk menangani output dari setiap layanan.
        thread = threading.Thread(target=stream_output, args=(process.stdout, name, service_color))
        thread.daemon = True # Thread akan mati jika skrip utama berhenti
        thread.start()
        
        print(f"-> Memulai: {name} (di folder ./{folder}) dengan warna.")
        time.sleep(1) # Beri jeda singkat antar start-up

    print("\n--- Semua layanan telah dimulai. Log akan muncul di bawah. ---")
    print("Tekan CTRL+C di terminal ini untuk menghentikan SEMUA layanan.")
    
    # Jaga agar skrip utama tetap berjalan
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n--- Menerima perintah berhenti (CTRL+C). Menghentikan semua layanan... ---")
finally:
    for process in processes:
        print(f"-> Menghentikan proses dengan PID: {process.pid}")
        process.terminate() # Mengirim sinyal untuk menghentikan setiap proses
    
    # Tunggu semua proses benar-benar berhenti
    for process in processes:
        process.wait()
        
    print("--- Semua layanan telah berhasil dihentikan. ---")