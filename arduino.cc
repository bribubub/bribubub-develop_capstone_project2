#include "WiFi.h"
#include "esp_camera.h"
#include <WebServer.h>

// === PENGATURAN WIFI ===
const char* ssid = "YARIS"; 
const char* password = "Sayang080624";

// === PIN RELAY PINTU ===
#define RELAY_PIN 4 

WebServer server(80);

// === KONFIGURASI PIN KAMERA AI-THINKER ===
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW); // Pintu terkunci dari awal

  WiFi.begin(ssid, password);
  Serial.println("\nKoneksi WiFi...");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.print("\n✅ WiFi Terhubung! IP KAMERA INI: ");
  Serial.println(WiFi.localIP()); 

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0; config.ledc_timer = LEDC_TIMER_0; config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM; config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM; config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM; config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM; config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM; config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM; config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM; config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM; config.xclk_freq_hz = 20000000; config.pixel_format = PIXFORMAT_JPEG;
  if(psramFound()){ config.frame_size = FRAMESIZE_VGA; config.jpeg_quality = 12; config.fb_count = 1; } else { config.frame_size = FRAMESIZE_SVGA; config.jpeg_quality = 12; config.fb_count = 1; }
  
  esp_camera_init(&config);

  // 1. RUTE UNTUK MEMOTRET (Dipanggil dari Python saat cek Wajah & Jarak)
  server.on("/scan", []() {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) { server.send(500, "text/plain", "Kamera Gagal"); return; }
    // Langsung kirim gambar mentah ke Python
    server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
  });

  // 2. RUTE UNTUK BUKA PINTU (Dipanggil dari Python kalau Wajah/Suara Benar)
  server.on("/open", []() {
    server.send(200, "text/plain", "OK");
    Serial.println("🔓 PERINTAH DITERIMA: MEMBUKA PINTU!");
    digitalWrite(RELAY_PIN, HIGH);
    delay(5000); // Buka pintu selama 5 detik
    digitalWrite(RELAY_PIN, LOW);
  });

  server.begin();
}

void loop() {
  server.handleClient(); 
}