---
title: Kurulum
description: anpr-pipeline'ı PyPI'dan kurun, Docker ile çalıştırın ya da uv ile kaynaktan hazırlayın.
---

anpr-pipeline Python 3.11 ile 3.14 arasında çalışır. Modeller ilk kullanımda indirilir (yaklaşık
30 MB) ve sonra önbellekte kalır.

## PyPI'dan

```bash
pip install anpr-pipeline
```

Bu, `anpr` komutunu kurar. Denemek için:

```bash
anpr version
```

### İsteğe bağlı ekler

| Ek | Ne getirir |
|---|---|
| `anpr-pipeline[gpu]` | Modelleri NVIDIA ekran kartında çalıştırmak için `onnxruntime-gpu`. |
| `anpr-pipeline[otel]` | API için OpenTelemetry izleme. Bkz. [İzleme](/tr/operations/monitoring/). |

## Docker ile

Depoda bir `Dockerfile` ve bir `docker-compose.yml` var. Servisin plakaları özetlemek için bir
gizli değere ihtiyacı var, buna pepper deniyor. Bir tane üretip servisi başlatın:

```bash
git clone https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR.git
cd Automatic_Number_Plate_Recognition_YOLO_OCR
export ANPR_PLATE_HMAC_PEPPER="$(python -c 'import secrets; print(secrets.token_hex(32))')"
docker compose up --build
```

<http://localhost:8000> adresini açın. Veritabanı `anpr-data` biriminde durur, yeniden başlatmada
kaybolmaz. Pepper'ı saklayın: başka bir pepper'la aynı plakanın özeti değişir. Nedeni için bkz.
[Gizlilik](/tr/operations/privacy/).

## Kaynaktan

[uv](https://docs.astral.sh/uv/) gerekir.

```bash
git clone https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR.git
cd Automatic_Number_Plate_Recognition_YOLO_OCR
uv sync --all-extras --group dev
cp .env.example .env
uv run anpr generate-pepper   # çıktıyı .env dosyasına ANPR_PLATE_HMAC_PEPPER olarak yapıştırın
uv run anpr serve
```

Sıradaki adım: [ilk plakanızı okuyun](/tr/start/first-plate/).
