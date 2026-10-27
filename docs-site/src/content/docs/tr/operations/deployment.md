---
title: Canlı ortama kurulum
description: anpr-pipeline'ı Docker, PostgreSQL, ters vekil sunucu ve ekran kartıyla canlı ortamda çalıştırın.
---

## Docker

İmaj `python:3.13-slim` üzerinde iki aşamada derlenir ve yetkisiz bir kullanıcıyla çalışır. Sağlık
denetimi `/health` adresini çağırır. `docker-compose.yml` tek sunucu için hazır bir kurulumdur:

```bash
export ANPR_PLATE_HMAC_PEPPER="…"   # `anpr generate-pepper` çıktısı, gizli değer deponuzda saklı
docker compose up -d --build
```

Compose, pepper olmadan başlamaz. SQLite veritabanı `anpr-data` biriminde durur. Etiketlenmiş her
sürüm ayrıca `ghcr.io/mftnakrsu/anpr-pipeline` adresine bir imaj yayınlar.

## Denetim listesi

- **Pepper'ı ayarlayın.** Gizli değer deponuzdan verin; yeniden başlatmalarda ve makineler arasında
  aynı kalsın.
- **Önüne bir ters vekil sunucu koyun.** API'de kimlik doğrulama yok. TLS'i ve erişimi nginx,
  Caddy, Traefik ya da ingress'iniz yönetsin; 8000 portunu doğrudan dışarı açmayın.
- **CORS'u daraltın.** `ANPR_CORS_ORIGINS` değerine yalnızca API'yi gerçekten çağıran kaynakları
  yazın.
- **Saklama süresini** amacınıza göre `ANPR_RETENTION_HOURS` ile seçin.
- **Günlükleri JSON yazın** (`ANPR_LOG_JSON=true`, Compose'da varsayılan), bir günlük toplayıcı
  okuyorsa.

## PostgreSQL

Tek makine için SQLite yeterli. Daha fazlası için PostgreSQL kullanın. Sürücü bağımlılıklar arasında
değil, paketin yanına kurun:

```bash
pip install anpr-pipeline asyncpg
export ANPR_DATABASE_URL="postgresql+asyncpg://anpr:parola@db:5432/anpr"
```

Tablo yoksa açılışta oluşturulur.

## Ekran kartı

OCR modeli, ONNX Runtime'ın CUDA sürümüyle NVIDIA ekran kartında çalışabilir:

```bash
pip install "anpr-pipeline[gpu]"
export ANPR_DEVICE=cuda
```

`ANPR_DEVICE=openvino`, kuruluysa Intel'in OpenVINO'sunu kullanır. `auto` hangisi varsa onu seçer.

## Bilinmesi gereken sınırlar

- Yüklemeler `ANPR_MAX_UPLOAD_MB`, akış kareleri `ANPR_MAX_WS_FRAME_BYTES`, ikisi birden de
  `ANPR_MAX_IMAGE_PIXELS` çözülmüş piksel ile sınırlıdır. Kameralarınızın çözünürlüğü düşükse
  belleği daha da sınırlamak için bu değerleri düşürün.
- Bir servis süreci her modelin bir kopyasını tutar. Çıkarım bir iş parçacığı havuzunda çalışır, bu
  yüzden tek süreç birden fazla akışa hizmet verir; sınır işlemcidir. Ölçeklemek için kapsayıcı
  sayısını artırın.

## Güvenlik sorunlarını bildirme

Lütfen herkese açık bir issue açmayın. GitHub'da
[özel bir güvenlik bildirimi](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/security/advisories/new)
kullanın;
[SECURITY.md](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/blob/main/SECURITY.md)
dosyasında anlatıldığı gibi.
