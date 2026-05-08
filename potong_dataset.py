import cv2
import os

# Ganti dengan nama file gambar yang kamu download
nama_file = 'dataset/ricarda/dataset_ricarda_grid.jpg'
img = cv2.imread(nama_file)

# Gambar kita punya 4 baris dan 3 kolom
tinggi, lebar, _ = img.shape
tinggi_potongan = tinggi // 4
lebar_potongan = lebar // 3

# Buat folder dataset otomatis
folder_tujuan = "dataset/ricarda"
os.makedirs(folder_tujuan, exist_ok=True)

hitung = 1
for baris in range(4):
    for kolom in range(3):
        # Potong gambar
        potongan = img[baris * tinggi_potongan : (baris + 1) * tinggi_potongan, 
                       kolom * lebar_potongan : (kolom + 1) * lebar_potongan]
        
        # Simpan file
        nama_simpan = os.path.join(folder_tujuan, f"ricarda_var_{hitung}.jpg")
        cv2.imwrite(nama_simpan, potongan)
        hitung = hitung + 1

print("✅ Selesai! 12 gambar dataset berhasil dipotong dan disimpan.")