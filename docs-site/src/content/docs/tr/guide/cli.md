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
Pepper ve API anahtarı `***` olarak görünür.

## `anpr version`

Kurulu sürümü basar.
