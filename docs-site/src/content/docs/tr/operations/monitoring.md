---
title: İzleme
description: anpr servisi için Prometheus metrikleri, yapılandırılmış günlükler ve OpenTelemetry izleme.
---

## Prometheus metrikleri

`GET /metrics`, metrikleri Prometheus metin biçiminde döndürür. Bir toplama görevini buraya
yönlendirin:

```yaml
scrape_configs:
  - job_name: anpr
    static_configs:
      - targets: ["anpr:8000"]
```

| Metrik | Tür | Etiketler | Anlamı |
|---|---|---|---|
| `anpr_requests_total` | counter | method, path, status | Karşılanan HTTP istekleri. |
| `anpr_request_seconds` | histogram | method, path | Bir HTTP isteğini yanıtlama süresi. |
| `anpr_inference_seconds` | histogram | route | `POST /api/v1/infer` üzerinde algılama ve OCR'a harcanan süre. |
| `anpr_detections_total` | counter | route, parsed | Okunan plakalar; `parsed` değeri `valid` ya da `invalid`. |
| `anpr_confirmed_plates_total` | counter | country, province_code | Akışta onaylanan plakalar; ülkeye ve Türkiye'de ile göre. Diğer ülkelerde `province_code` boştur. `country` etiketi 0.3.0'da eklendi. |
| `anpr_active_websocket_streams` | gauge | | Açık `/ws/stream` bağlantıları. |
| `anpr_watchlist_matches_total` | counter | source | [İzleme listesinde](/tr/guide/watchlist/) bulunan onaylı plakalar; `source` değeri `stream` ya da `watch`. |
| `anpr_webhook_deliveries_total` | counter | result | Yeniden denemelerden sonra webhook çağrıları; `result` değeri `ok` ya da `failed`. |

Bir panoya iki sayı koymaya değer: `anpr_detections_total` içinde `invalid` payı (kamera açısı ya
da odak bozulunca yükselir) ve `anpr_inference_seconds` değerinin 95. yüzdeliği.

## Günlükler

Günlükleri structlog yazar. Varsayılan olarak metindir; `ANPR_LOG_JSON=true` ile her satır Loki,
Elasticsearch ya da CloudWatch'a hazır bir JSON nesnesi olur. Bir istek işlenirken yazılan her
günlük satırı o isteğin `request_id`, `method` ve `path` değerlerini taşır. Yanıt da aynı kimliği
`X-Request-ID` başlığında döndürür; böylece başarısız bir çağrı günlüklerde bulunabilir.

Uyarı kurmaya değer olaylar:

| Olay | Anlamı |
|---|---|
| `infer.image_rejected`, `stream.frame_rejected` | Bir resim ya da kare boyut sınırlarını aştı. Kısa sürede çok sayıda gelmesi, birinin servisi yokladığı anlamına gelebilir. |
| `stream.error` | Bir akış beklenmedik bir hatayla bitti. |
| `storage.retention.error` | Saklama süresi görevi eski satırları silemedi. |

## OpenTelemetry

Eki kurun ve bir OTLP adresi verin; servis her isteği izler:

```bash
pip install "anpr-pipeline[otel]"
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
export OTEL_SERVICE_NAME=anpr   # varsayılan
anpr serve
```

Adres ya da ek olmadan izleme kapalı kalır ve hiçbir yük getirmez.
