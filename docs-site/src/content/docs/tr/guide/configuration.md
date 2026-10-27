---
title: Yapılandırma
description: Bütün ANPR_ ortam değişkenleri, varsayılanları ve neyi değiştirdikleri.
---

Bütün ayarlar `ANPR_` ile başlayan ortam değişkenleridir. Servisi başlattığınız klasördeki bir
`.env` dosyasına da yazılabilirler; depodaki `.env.example` hepsini listeler. Ayarlar açılışta bir
kez okunur, bir ayarı değiştirdikten sonra servisi yeniden başlatın. Geçerli değerleri
`anpr dump-config` gösterir.

## Kayıt ve gizlilik

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `ANPR_PLATE_HMAC_PEPPER` | rastgele, uyarıyla | Plakaların özetlendiği gizli değer. En az 32 karakter; `anpr generate-pepper` bir tane üretir. **Canlı ortamda mutlaka ayarlayın**: rastgele pepper her yeniden başlatmada değişir ve aynı plakanın özeti de değişir. |
| `ANPR_DATABASE_URL` | `sqlite+aiosqlite:///data/anpr.db` | Algılamaların kaydedildiği yer. PostgreSQL için `asyncpg` kurun ve `postgresql+asyncpg://kullanıcı:parola@sunucu/veritabanı` kullanın. |
| `ANPR_RETENTION_HOURS` | `720` (30 gün) | Bundan eski satırlar silinir. Denetim açılışta ve sonra her saat çalışır. |

## İşlem hattı

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `ANPR_DETECTOR_MODEL` | `yolo-v9-t-384-license-plate-end2end` | fast-alpr dedektör modelinin adı ya da kendi ONNX dosyanızın yolu. |
| `ANPR_DETECTOR_CONF_THRESHOLD` | `0.4` | Dedektörün bundan daha az emin olduğu kutular atılır. |
| `ANPR_OCR_MODEL` | `cct-s-v2-global-model` | fast-plate-ocr modelinin adı. |
| `ANPR_DEVICE` | `auto` | OCR modelinin nerede çalışacağı: `auto`, `cpu`, `cuda` ya da `openvino`. |
| `ANPR_DETECT_EVERY_N_FRAMES` | `3` | WebSocket akışında yalnızca her N. kareyi işler. |
| `ANPR_MIN_TRACK_DWELL` | `5` | Bir plakanın onaylanmadan önce okunması gereken işlenmiş kare sayısı. |

`ANPR_SR_MIN_PLATE_WIDTH` ve `ANPR_OCR_FALLBACK_THRESHOLD` okunur ama henüz hiçbir yerde
kullanılmaz; süper çözünürlük ve ikinci bir OCR motoru için ayrılmışlardır. Bkz.
[Yol haritası](/tr/roadmap/).

## API

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `ANPR_API_HOST` | `0.0.0.0` | `anpr serve` komutunun dinlediği adres. |
| `ANPR_API_PORT` | `8000` | `anpr serve` komutunun dinlediği port. |
| `ANPR_MAX_UPLOAD_MB` | `10` | `POST /api/v1/infer` uç noktasının kabul ettiği en büyük dosya. |
| `ANPR_MAX_IMAGE_PIXELS` | `40000000` | Hem yüklemede hem akışta çözülmüş resmin en fazla piksel sayısı. 40 MP, 8K'nın üstünde. |
| `ANPR_MAX_WS_FRAME_BYTES` | `4194304` (4 MB) | `/ws/stream` üzerindeki en büyük kare. |
| `ANPR_CORS_ORIGINS` | `*` | API'yi tarayıcıdan çağırabilecek kaynaklar, virgülle ayrılmış. Canlı ortamda daraltın. |

## Günlükler ve izleme

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `ANPR_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` ya da `ERROR`. |
| `ANPR_LOG_JSON` | `false` | Loki, Elasticsearch ve benzerleri için JSON günlük satırları. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | ayarsız | OpenTelemetry izlemeyi açar. `otel` eki gerekir. |
| `OTEL_SERVICE_NAME` | `anpr` | İzlerdeki servis adı. |
