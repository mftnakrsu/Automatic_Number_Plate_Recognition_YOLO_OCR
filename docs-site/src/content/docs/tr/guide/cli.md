---
title: Komut satırı
description: anpr komutunun bütün alt komutları ve seçenekleri.
---

`anpr` komutu paketle birlikte gelir. `anpr --help` alt komutları listeler, `anpr <komut> --help`
bir komutun seçeneklerini gösterir.

## Genel seçenekler

Bunlar komuttan önce yazılır: `anpr --log-json serve` gibi.

| Seçenek | Ortam değişkeni | Varsayılan | Anlamı |
|---|---|---|---|
| `--log-level` | `ANPR_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` ya da `ERROR`. |
| `--log-json` | `ANPR_LOG_JSON` | kapalı | Günlükleri renkli metin yerine JSON satırları olarak yazar. |

## `anpr infer RESİM`

Tek bir resim dosyasındaki plakaları bulur ve okur. Tablo basar, `--json` ile JSON. Sütunların
anlamı için bkz. [İlk plakanız](/tr/start/first-plate/).

| Seçenek | Anlamı |
|---|---|
| `--json` | Sonucu tablo yerine JSON dizisi olarak basar. |

Dosya yoksa 2 koduyla çıkar. Hiçbir şey kaydetmez.

## `anpr watch KAYNAK`

Videodan plaka okur: bir dosyadan, bir ağ yayınından ya da bir kameradan.
[WebSocket akışındaki](/tr/guide/websocket/) kare atlama, takip ve oylamanın aynısını çalıştırır ve
her plakayı onaylandığında bir kez basar. 0.3.0'dan beri.

```bash
anpr watch trafik.mp4
anpr watch rtsp://kullanici:parola@192.168.1.20:554/stream1
anpr watch 0                       # ilk yerel kamera
anpr watch trafik.mp4 --json       # her satıra bir JSON nesnesi
```

```text
frame     42  track    3  34 ABC 1234  İstanbul  det=0.91
frame    118  track    7  06 AA 42  Ankara  det=0.88
240 frames read, 2 plates confirmed.
```

| Seçenek | Anlamı |
|---|---|
| `--json` | Onaylanan her plaka için bir JSON nesnesi basar: `frame_idx`, `track_id`, `plate`, `province_code`, `province_name`, `confidence`, `bbox` ve `watchlist` ([izleme listesiyle](/tr/guide/watchlist/) eşleşmede `{"label": ...}`, yoksa `null`). |
| `--store` | Onaylanan her plakayı, akışın yaptığı gibi özetleyerek `ANPR_DATABASE_URL` adresine de kaydeder. Önce `ANPR_PLATE_HMAC_PEPPER` ayarlayın. |
| `--max-frames N` | N kareden sonra durur. |
| `--reconnect N` | Yayın ve kameralarda bağlantı kopunca kaynağın kaç kez yeniden açılacağı (varsayılan 5, iki saniye arayla). Dosya ise biter. |

`ANPR_DETECT_EVERY_N_FRAMES` ve `ANPR_MIN_TRACK_DWELL` akıştaki gibi geçerlidir. Özet satırı
standart hataya yazılır, böylece `--json` çıktısı olduğu gibi başka bir komuta aktarılabilir.
`Ctrl+C` durdurur. Kaynak açılamazsa komut 2 koduyla çıkar.

`ANPR_WEBHOOK_URL` ayarlıyken ya da `--store` ile onaylanan plakalar
[izleme listesinde](/tr/guide/watchlist/) de aranır; eşleşme webhook'u çağırır.

## `anpr watchlist`

Uyarı verecek plakaları yönetir: `add`, `import`, `remove` ve `list`. Bkz.
[İzleme listesi ve webhook](/tr/guide/watchlist/#komutlar). 0.3.0'dan beri.

## `anpr serve`

HTTP ve WebSocket servisini uvicorn ile başlatır.

| Seçenek | Ortam değişkeni | Varsayılan | Anlamı |
|---|---|---|---|
| `--host` | `ANPR_API_HOST` | `0.0.0.0` | Dinlenecek adres. |
| `--port` | `ANPR_API_PORT` | `8000` | Dinlenecek port. |
| `--reload` | | kapalı | Kod değişince yeniden başlatır. Yalnızca geliştirme için. |

## `anpr generate-pepper`

`ANPR_PLATE_HMAC_PEPPER` için 64 karakterlik yeni, rastgele bir onaltılık dizi basar.

## `anpr generate-api-key`

`ANPR_API_KEY` için yeni, rastgele bir anahtar basar. 0.2.5'ten beri.

## `anpr dump-config`

Ortam değişkenleri ve `.env` okunduktan sonra servisin kullanacağı ayarları JSON olarak basar.
Pepper, API anahtarı, webhook adresi ve gizli değeri `***` olarak görünür.

## `anpr version`

Kurulu sürümü basar.
