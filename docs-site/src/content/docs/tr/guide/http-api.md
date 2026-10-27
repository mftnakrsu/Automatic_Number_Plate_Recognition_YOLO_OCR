---
title: HTTP API
description: anpr servisinin REST uç noktaları, istek ve yanıt örnekleriyle.
---

`anpr serve` bir FastAPI uygulaması başlatır. Aşağıdaki örnekler servisin `localhost:8000`
adresinde çalıştığını varsayıyor. FastAPI ayrıca `/docs` adresinde etkileşimli bir belge,
`/openapi.json` adresinde de OpenAPI şemasını sunar.

:::caution
Uç noktalarda kimlik doğrulama yok. Servisi bu hâliyle internete açmayın; erişimi denetleyen bir
ters vekil sunucunun (reverse proxy) arkasına koyun ya da özel bir ağda tutun. Bkz.
[Canlı ortama kurulum](/tr/operations/deployment/).
:::

## `POST /api/v1/infer`

Tek bir resimdeki plakaları okur. Resmi `file` adlı bir multipart form alanı olarak gönderin.
Pillow'un açabildiği her biçim olur: JPEG, PNG, WebP, BMP ve benzerleri.

```bash
curl -F file=@araba.jpg http://localhost:8000/api/v1/infer
```

Yanıt, bulunan her plaka için bir öğe içeren bir listedir. Plaka yoksa boş liste döner:

```json
[
  {
    "bbox": { "x1": 412, "y1": 388, "x2": 618, "y2": 436 },
    "detection_confidence": 0.91,
    "raw_text": "34ABC1Z34",
    "normalized_text": "34ABC1234",
    "parsed": {
      "province_code": "34",
      "province_name": "İstanbul",
      "letters": "ABC",
      "digits": "1234",
      "canonical": "34 ABC 1234"
    }
  }
]
```

| Alan | Anlamı |
|---|---|
| `bbox` | Plakanın kutusu, piksel cinsinden. |
| `detection_confidence` | Dedektörün ne kadar emin olduğu, 0 ile 1 arası. |
| `raw_text` | OCR modelinin okuduğu metin. Hiçbir şey okuyamadıysa `null`. |
| `normalized_text` | Boşlukları atılmış, büyük harfe çevrilmiş ve benzer karakterleri düzeltilmiş `raw_text`. |
| `parsed` | Okumanın il, harf ve rakam olarak bölünmüş hâli. Geçerli bir Türk plakası değilse `null`. |

Bu uç nokta hiçbir şey kaydetmez. Onaylanan plakaları yalnızca [WebSocket akışı](/tr/guide/websocket/)
kaydeder.

### Hatalar

| Durum kodu | Ne zaman |
|---|---|
| `400` | Yükleme boşsa (`empty upload`) ya da resim olarak çözülemiyorsa (`cannot decode image`). |
| `413` | Dosya `ANPR_MAX_UPLOAD_MB` değerinden büyükse (`upload too large`) ya da çözülmüş boyutu `ANPR_MAX_IMAGE_PIXELS` değerini aşıyorsa (`image too large`). |
| `422` | İstekte `file` alanı yoksa. |

Piksel sınırı, resim çözülmeden önce dosya başlığından denetlenir. Böylece kendini dev gibi
gösteren küçük bir dosya bellek harcatmadan geri çevrilir.

## `GET /api/v1/detections`

Kaydedilmiş son plakaları en yeniden başlayarak listeler.

```bash
curl "http://localhost:8000/api/v1/detections?limit=20&confirmed_only=true"
```

| Sorgu parametresi | Varsayılan | Anlamı |
|---|---|---|
| `limit` | `100` | Kaç satır döneceği, 1 ile 1000 arası. |
| `confirmed_only` | `false` | Yalnızca onaylanmış okumaları döndürür. |

```json
[
  {
    "id": 17,
    "timestamp": "2026-10-27T09:41:12.503000",
    "track_id": 3,
    "plate_hmac": "9f2c…e41a",
    "province_code": "34",
    "confidence": 0.88,
    "bbox": [412, 388, 618, 436],
    "confirmed": true
  }
]
```

Yanıtta plaka metni yok, çünkü veritabanında da yok. Bilinen bir plakanın görülüp görülmediğini
anlamak için aynı pepper'la özetini hesaplayıp `plate_hmac` ile karşılaştırın; bkz.
[Gizlilik](/tr/operations/privacy/).

## Servis uç noktaları

| Uç nokta | Yanıt |
|---|---|
| `GET /health` | Uygulama ayaktaysa `{"status": "ok"}`. Canlılık denetimi (liveness probe) için kullanın. |
| `GET /version` | `{"version": "0.2.4"}` |
| `GET /metrics` | Prometheus metin biçiminde metrikler. Bkz. [İzleme](/tr/operations/monitoring/). |
| `GET /` | Resim yükleme ve web kamerası akışı olan web sayfası. |

Her yanıtta bir `X-Request-ID` başlığı bulunur. İstek bu başlığı gönderdiyse aynen geri döner,
göndermediyse servis bir tane üretir. Aynı kimlik günlüklerde de görünür.
