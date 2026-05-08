import os, sys, time, threading, wave
import smtplib, random
from email.message import EmailMessage

# --- AUTO VIRTUAL ENVIRONMENT SWITCHER ---
# Mengecek apakah script dijalankan di luar Virtual Environment (venv)
if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
    # Jika di luar venv, otomatis cari venv/bin/python dan jalankan ulang
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'bin', 'python')
    if os.path.exists(venv_python):
        print("🔄 Otomatis mengaktifkan Virtual Environment (venv)...")
        os.execv(venv_python, [venv_python] + sys.argv)
    else:
        print("⚠️ Peringatan: Folder 'venv' tidak ditemukan! Pastikan sudah menginstall library.")
# -----------------------------------------

import cv2
import numpy as np
import requests
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template, redirect, Response
from deepface import DeepFace

try:
    import boto3
    from botocore.client import Config
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    print("⚠️ Boto3 tidak terinstall, gunakan 'pip install boto3'")

app = Flask(__name__)

# ─────────────────────────────────────────────
# ⚙️ CONFIG & HARDWARE IP (Ubah sesuai IP di Serial Monitor Arduino)
# ─────────────────────────────────────────────
ESP32_CAM_IP     = "http://192.168.8.120"  # IP ESP32 Kamera (Hanya untuk /scan)
ESP32_MIC_IP     = "http://192.168.0.118"  # Kembalikan ini ke aslinya (atau ubah jika ada IP MIC khusus)
CLOUD_API_URL    = "https://api.testrumahcloud.site/api/access-log"

# --- SUPABASE CONFIGURATION (S3) ---
SUPABASE_S3_ENDPOINT = "https://lqyjvssuwnidxebfrrlc.supabase.co/storage/v1/s3"
SUPABASE_S3_ACCESS_KEY = "0de326ba854b5a5cbafbc7140bc0627c"
SUPABASE_S3_SECRET_KEY = "410da4c221a179f7aab9d5c83d076891c7703d1b009dbc350c2e154013754510"
SUPABASE_BUCKET = "dataset"

# --- SUPABASE DATABASE (REST API) & EMAIL OTP CONFIG ---
SUPABASE_REST_URL = "https://lqyjvssuwnidxebfrrlc.supabase.co"
SUPABASE_API_KEY = "0de326ba854b5a5cbafbc7140bc0627c" # Harus diisi agar bisa cek email
SUPABASE_TABLE_USERS = "users" # Nama tabel penyimpanan email di DB Supabase

SENDER_EMAIL = "ricardasebastian95@gmail.com" # Email pengirim
SENDER_PASSWORD = "vbiz ncpg qjbi nemj"   # Gunakan App Password dari Google (bukan password asli email)

s3_client = None
if HAS_BOTO3:
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=SUPABASE_S3_ENDPOINT,
            aws_access_key_id=SUPABASE_S3_ACCESS_KEY,
            aws_secret_access_key=SUPABASE_S3_SECRET_KEY,
            region_name='us-east-1',
            config=Config(s3={'addressing_style': 'path'}, signature_version='s3v4')
        )
        print("✅ Terhubung ke Supabase (via S3)!")
    except Exception as e:
        print(f"❌ Gagal koneksi Supabase S3: {e}")

# Konfigurasi AI DeepFace
MODEL_NAME       = "ArcFace"       
DETECTOR_BACKEND = "retinaface"    
DISTANCE_METRIC  = "cosine"        
THRESHOLD        = 0.40            
BLUR_THRESHOLD   = 5.0            
MIN_BRIGHTNESS   = 40              
MAX_BRIGHTNESS   = 230             
DATASET_DIR      = "dataset"
STATIC_DIR       = "static"

# ─────────────────────────────────────────────
# 🧠 IN-MEMORY STORE & STATES
# ─────────────────────────────────────────────
known_embeddings: dict[str, list] = {}
failed_attempts = 0
current_otp_str = ""

# --- ALIGNMENT STATE ---
# Inisialisasi Haar Cascade untuk mendeteksi wajah di web UI
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
latest_alignment = {"color": "red", "status": "Mendeteksi Wajah..."}


# ─────────────────────────────────────────────
# 🖼️ IMAGE PREPROCESSING & QUALITY CHECK
# ─────────────────────────────────────────────
def preprocess_image(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 6, 6, 7, 21)
    return denoised

def check_image_quality(img: np.ndarray) -> tuple[bool, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < BLUR_THRESHOLD:
        return False, f"Terlalu buram (score: {blur_score:.1f})"
    
    brightness = gray.mean()
    if brightness < MIN_BRIGHTNESS:
        return False, f"Terlalu gelap (brightness: {brightness:.1f})"
    if brightness > MAX_BRIGHTNESS:
        return False, f"Terlalu silau (brightness: {brightness:.1f})"
    return True, "OK"

# ─────────────────────────────────────────────
# 🤖 AI TRAINING (MEMUAT WAJAH SAAT SERVER NYALA)
# ─────────────────────────────────────────────
def get_embedding(img: np.ndarray) -> np.ndarray | None:
    try:
        result = DeepFace.represent(
            img_path=img, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND,
            enforce_detection=True, align=True
        )
        if result and len(result) > 0:
            return np.array(result[0]["embedding"])
    except Exception:
        pass
    return None

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(1 - np.dot(a_norm, b_norm))

def sync_dataset_from_supabase():
    if not s3_client: return
    print("\n☁️ Sinkronisasi dataset dari Supabase ke lokal...")
    try:
        # Menggunakan S3 untuk mengambil daftar file
        response = s3_client.list_objects_v2(Bucket=SUPABASE_BUCKET)
        if 'Contents' in response:
            for obj in response['Contents']:
                file_key = obj['Key']
                
                # Cek apakah ini file di dalam folder (misal: "ricarda/foto.jpg")
                if '/' in file_key:
                    parts = file_key.split('/')
                    if len(parts) >= 2 and parts[-1] != "":
                        name = parts[0]
                        file_name = parts[-1]
                        
                        user_dir = os.path.join(DATASET_DIR, name)
                        os.makedirs(user_dir, exist_ok=True)
                        
                        local_path = os.path.join(user_dir, file_name)
                        # Hanya download jika belum ada di lokal
                        if not os.path.exists(local_path):
                            print(f"  📥 Mendownload {file_name} untuk {name}...")
                            s3_client.download_file(SUPABASE_BUCKET, file_key, local_path)
        print("✅ Sinkronisasi Supabase selesai!")
    except Exception as e:
        print(f"⚠️ Peringatan: Gagal sinkronisasi Supabase S3: {e}")

def train_faces():
    global known_embeddings
    known_embeddings = {}
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Sinkronisasi dari cloud sebelum melatih AI
    sync_dataset_from_supabase()
    
    print("\n📚 Memuat dataset wajah ke dalam memori AI...")

    for name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(folder_path): continue

        known_embeddings[name] = []
        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None: continue

            good, reason = check_image_quality(img)
            if not good:
                print(f"  ⚠️  Melewati {file} [{name}]: {reason}")
                continue

            processed = preprocess_image(img)
            embedding = get_embedding(processed)

            if embedding is not None:
                known_embeddings[name].append(embedding)

    known_embeddings = {k: v for k, v in known_embeddings.items() if len(v) > 0}
    print(f"🎉 Selesai! Wajah yang siap dikenali: {list(known_embeddings.keys())}\n")

# Panggil fungsi training
train_faces()

# ─────────────────────────────────────────────
# 🔍 RECOGNITION LOGIC (MENCOCOKKAN WAJAH)
# ─────────────────────────────────────────────
def recognize_face(img: np.ndarray) -> dict:
    result = {"nama": "unknown", "akses": False, "confidence": 0.0, "reason": "Wajah tidak terdeteksi"}
    if not known_embeddings: return result

    good, reason = check_image_quality(img)
    if not good:
        result["reason"] = reason
        return result

    processed = preprocess_image(img)
    query_embedding = get_embedding(processed)
    if query_embedding is None:
        result["reason"] = "Gagal mengekstrak struktur wajah"
        return result

    best_name = "unknown"
    best_distance = float("inf")

    for name, embeddings in known_embeddings.items():
        for stored_embedding in embeddings:
            dist = cosine_distance(query_embedding, stored_embedding)
            if dist < best_distance:
                best_distance = dist
                best_name = name

    confidence = max(0.0, (1.0 - (best_distance / THRESHOLD))) * 100.0
    if best_distance <= THRESHOLD:
        result.update({"nama": best_name, "akses": True, "confidence": round(confidence, 1), "reason": "Wajah Dikenali!"})
    else:
        result.update({"nama": "unknown", "akses": False, "confidence": round(confidence, 1), "reason": "Bukan penghuni rumah"})
    return result

# ─────────────────────────────────────────────
# ☁️ CLOUD API SENDER (PENYALUR KE DATABASE)
# ─────────────────────────────────────────────
def kirim_log_ke_cloud(nama, metode, akses):
    status_api = "GRANTED" if akses else "DENIED"
    
    user_id = 99
    nama_lower = nama.lower()
    if "budi" in nama_lower: user_id = 1
    elif "siti" in nama_lower: user_id = 2
    elif "andi" in nama_lower: user_id = 3
    elif "eko" in nama_lower: user_id = 4
    elif "ricarda" in nama_lower or "richard" in nama_lower: user_id = 10
    
    payload = {
        "accessType": metode, "status": status_api, "deviceId": 1, "userId": user_id
    }
    try:
        print(f"☁️ Mengirim Log API: {payload}")
        res = requests.post(CLOUD_API_URL, json=payload, timeout=5)
        if res.status_code in [200, 201]: print("✅ SUKSES: Masuk ke Web Temanmu!")
        else: print(f"⚠️ Gagal API: {res.status_code}")
    except Exception as e:
        print(f"❌ Error Jaringan Cloud: {e}")

# ─────────────────────────────────────────────
# 🎥 LIVE VIDEO STREAMING (MJPEG UNTUK WEB)
# ─────────────────────────────────────────────
def gen_frames():
    global latest_alignment
    while True:
        try:
            resp = requests.get(f"{ESP32_CAM_IP}/scan", timeout=2)
            frame = resp.content
            
            # --- Cek Alignment Wajah secara Real-time ---
            try:
                nparr = np.frombuffer(frame, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # minNeighbors=6 agar lebih strict mencegah false positive
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
                    
                    if len(faces) > 0:
                        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                        x, y, w, h = faces[0]
                        
                        height, width = img.shape[:2]
                        center_x, center_y = width // 2, height // 2
                        face_center_x, face_center_y = x + w//2, y + h//2
                        
                        dist_x = abs(center_x - face_center_x)
                        dist_y = abs(center_y - face_center_y)
                        
                        # Syarat Hijau (Pas di tengah dan cukup dekat)
                        if dist_x < width * 0.25 and dist_y < height * 0.25 and h > height * 0.3:
                            latest_alignment = {"color": "green", "status": "Wajah Pas! Tahan 2 detik..."}
                        # Syarat Oranye (Hampir pas)
                        elif dist_x < width * 0.4 and dist_y < height * 0.4 and h > height * 0.2:
                            latest_alignment = {"color": "orange", "status": "Sedikit lagi... Arahkan wajah ke tengah"}
                        else:
                            latest_alignment = {"color": "red", "status": "Wajah belum pas"}
                    else:
                        latest_alignment = {"color": "red", "status": "Wajah tidak terdeteksi"}
            except Exception as e:
                print("Error di cek alignment:", e)
                latest_alignment = {"color": "red", "status": "Memproses kamera..."}
            # -------------------------------------------
            
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.2) 
        except:
            time.sleep(1)

# ─────────────────────────────────────────────
# 🌐 FLASK ROUTES (ANTARMUKA WEB & KONTROL LOGIKA)
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alignment')
def get_alignment():
    return jsonify(latest_alignment)

@app.route('/recognize', methods=['POST'])
def route_recognize():
    global failed_attempts
    try:
        print("\n📸 Mengambil foto untuk AI...")
        resp = requests.get(f"{ESP32_CAM_IP}/scan", timeout=5)
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)

        hasil = recognize_face(img)
        print(f"🧠 Keputusan AI: {hasil}")

        # Pintu hanya dibuka jika akses True
        if hasil["akses"]:
            failed_attempts = 0 # Reset kegagalan
            print("🔓 MEMBERIKAN PERINTAH BUKA RELAY...")
            try: requests.get(f"{ESP32_MIC_IP}/open", timeout=5) 
            except: pass
        else:
            # Hanya tambahkan failed_attempts jika wajah terdeteksi tapi ditolak,
            # jangan tambah jika memang kosong / error gambar.
            alasan_kegagalan = str(hasil.get("reason", "")).lower()
            if "tidak terdeteksi" not in alasan_kegagalan and "gagal mengekstrak" not in alasan_kegagalan and "buram" not in alasan_kegagalan and "gelap" not in alasan_kegagalan and "silau" not in alasan_kegagalan:
                failed_attempts += 1
                print(f"⚠️ Gagal mengenali wajah. Percobaan gagal ke-{failed_attempts}")
                if failed_attempts >= 3:
                    hasil["switch_to_voice"] = True
                    hasil["reason"] = "Akses Ditolak 3 Kali. Silakan Verifikasi Email!"
                    failed_attempts = 0 # Reset biar tidak looping terus menerus di angka 3
                    return jsonify(hasil)
            else:
                print(f"⚠️ Melewati gambar: {hasil.get('reason')}. Tidak menambah failed attempts.")

        # Selalu catat log siapa pun yang mencoba masuk
        kirim_log_ke_cloud(hasil["nama"], "Face Recognition", hasil["akses"])
        
        # Update status global untuk diakses oleh /status
        global last_status
        last_status = hasil
        
        return jsonify(hasil)
    except Exception as e:
        return jsonify({"error": str(e), "akses": False})

# Status Global
last_status = {"reason": "Sistem Siap. Sedang memantau area pintu...", "akses": False}

@app.route('/status')
def get_status():
    return jsonify(last_status)

# --- OTP & SUPABASE VERIFICATION ROUTE ---
def send_otp_email_thread(email_penerima, otp_code):
    try:
        msg = EmailMessage()
        msg.set_content(f"Kode OTP Anda untuk masuk adalah: {otp_code}\n\nSilakan sebutkan kode ini di depan sensor mikrofon (contoh pengucapan: 'Satu Dua Tiga').")
        msg['Subject'] = 'Kode OTP Smart Door'
        msg['From'] = SENDER_EMAIL
        msg['To'] = email_penerima

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"✅ OTP {otp_code} berhasil dikirim ke {email_penerima}")
    except Exception as e:
        print(f"❌ Gagal mengirim email OTP: {e}")

@app.route('/request_otp', methods=['POST'])
def request_otp():
    global current_otp_str
    try:
        data = request.get_json()
        email_input = data.get("email") if data else None
        
        if not email_input:
            return jsonify({"status": "gagal", "pesan": "Email tidak boleh kosong!"})
            
        print(f"🔍 Membuat OTP dan mengirimkan ke email: {email_input}...")
        
        # Buat OTP 3 digit
        current_otp_str = str(random.randint(100, 999))
        
        # Kirim email asinkron agar server tidak tersendat
        threading.Thread(target=send_otp_email_thread, args=(email_input, current_otp_str)).start()
        
        return jsonify({"status": "sukses", "pesan": "OTP sedang dikirim ke email Anda!"})
            
    except Exception as e:
        print(f"Error Request OTP: {e}")
        return jsonify({"status": "gagal", "pesan": f"Error server: {str(e)}"})

# --- RUTE UJI COBA (DUMMY API) ---
@app.route('/register_face', methods=['POST'])
def register_face():
    try:
        nama = request.form.get('nama')
        file = request.files.get('file')

        if not nama or not file:
            return jsonify({"status": "gagal", "pesan": "Nama atau file tidak boleh kosong!"})

        # Sanitize nama folder
        nama = "".join(c for c in nama if c.isalnum() or c in (" ", "_", "-")).strip()
        if not nama:
            return jsonify({"status": "gagal", "pesan": "Nama tidak valid!"})

        # Bikin folder berdasarkan nama
        user_dir = os.path.join(DATASET_DIR, nama)
        os.makedirs(user_dir, exist_ok=True)

        # Simpan file
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            return jsonify({"status": "gagal", "pesan": "Format file hanya JPG/PNG!"})

        # Penamaan file unik dengan timestamp
        filename = f"{nama}_{int(time.time())}{file_ext}"
        filepath = os.path.join(user_dir, filename)
        file.save(filepath)

        print(f"\n📥 Foto baru disimpan untuk '{nama}' di {filepath}")
        
        # Upload ke Supabase jika dikonfigurasi
        if s3_client:
            print(f"☁️ Mengupload {filename} ke Supabase S3...")
            content_type = "image/png" if file_ext == ".png" else "image/jpeg"
            try:
                s3_client.upload_file(
                    filepath, 
                    SUPABASE_BUCKET, 
                    f"{nama}/{filename}",
                    ExtraArgs={'ContentType': content_type}
                )
                print("✅ Berhasil upload ke Supabase S3!")
            except Exception as e:
                print(f"⚠️ Gagal upload ke Supabase S3: {e}")
        
        # Latih ulang AI
        print("🔄 Memulai ulang pelatih wajah...")
        threading.Thread(target=train_faces).start()

        return jsonify({"status": "sukses", "pesan": f"Wajah '{nama}' berhasil didaftarkan!"})
    except Exception as e:
        print(f"❌ Error Register Face: {e}")
        return jsonify({"status": "gagal", "pesan": f"Error: {str(e)}"})

# --- RUTE UJI COBA (DUMMY API) ---
@app.route('/test_dummy_api')
def test_dummy_api():
    print("\n🧪 Menjalankan simulasi API Dummy (Ricarda/Richard)...")
    kirim_log_ke_cloud("ricarda", "Face Recognition", True)
    return jsonify({"status": "OK", "pesan": "Dummy berhasil ditembak! Cek database Hoppscotch."})

# ─────────────────────────────────────────────
# 🎙️ ROUTE SUARA (VOICE COMMAND & AI GOOGLE)
# ─────────────────────────────────────────────
DATA_DIR = "dataset_suara"
os.makedirs(DATA_DIR, exist_ok=True)
recording = False
frames = []

def stream_audio():
    global frames, recording
    try:
        r = requests.get(f"{ESP32_MIC_IP}/stream", stream=True, timeout=5)
        for chunk in r.iter_content(chunk_size=1024):
            if not recording: break
            if chunk: frames.append(chunk)
        r.close()
    except Exception as e: print(f"Error Mic: {e}")

@app.route('/start_voice')
def start_voice():
    global recording, frames; recording, frames = True, []
    threading.Thread(target=stream_audio).start()
    return "OK"

@app.route('/stop_voice')
def stop_voice():
    global recording, frames; recording = False; time.sleep(1)
    
    fname = f"{DATA_DIR}/v_{int(time.time())}.wav"
    with wave.open(fname, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    r = sr.Recognizer()
    try:
        with sr.AudioFile(fname) as source: audio = r.record(source)
        teks = r.recognize_google(audio, language="id-ID").lower()
        print(f"🎙️ Kata Sandi Tertangkap: {teks}")
        
        global current_otp_str
        
        # Konversi kata ke angka untuk memudahkan pencocokan
        kata_angka = {"satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5", "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9", "nol": "0", "kosong": "0"}
        teks_angka = teks
        for k, v in kata_angka.items():
            teks_angka = teks_angka.replace(k, v)
        teks_angka = teks_angka.replace(" ", "") # Gabungkan, misal "1 2 3" -> "123"
        
        is_valid = False
        # Jika ada OTP aktif, cocokkan dengan OTP
        if current_otp_str and current_otp_str in teks_angka:
            is_valid = True
            current_otp_str = "" # Reset agar tidak bisa dipakai ulang
        # Fallback darurat jika sedang development/tidak pakai OTP
        elif not current_otp_str and ("123" in teks_angka):
            is_valid = True
            
        if is_valid:
            requests.get(f"{ESP32_MIC_IP}/open", timeout=5)
            kirim_log_ke_cloud("ricarda", "Voice Command", True)
            return jsonify({"status": "sukses", "pesan": f"🔓 SANDI BENAR! ({teks})"})
        else:
            kirim_log_ke_cloud("unknown", "Voice Command", False)
            return jsonify({"status": "gagal", "pesan": f"🔒 SANDI SALAH! ({teks})"})
    except:
        return jsonify({"status": "gagal", "pesan": "❌ SUARA TIDAK TERDENGAR JELAS"})

# ─────────────────────────────────────────────
# 🚀 EKSEKUSI SERVER
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 SERVER SMART DOOR SIAP BEROPERASI")
    print("="*50)
    print("Silakan buka browser di alamat: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)