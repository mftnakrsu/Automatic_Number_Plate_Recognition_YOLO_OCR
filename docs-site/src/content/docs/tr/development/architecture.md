---
title: Mimari
description: Bir karenin nasıl kayıtlı bir özete dönüştüğü ve her adımı hangi modülün yaptığı.
---

```text
Tek resim (POST /api/v1/infer, anpr infer)

  çöz ─▶ algıla ─▶ kırp ─▶ OCR ─▶ biçim denetimi + benzer karakter düzeltmesi ─▶ JSON

Akış (/ws/stream)

  çöz ─▶ her N. kare ─▶ algıla ─▶ izle ─▶ kırp + boyutlandır ─▶ OCR
      ─▶ biçim denetimi + benzer karakter düzeltmesi ─▶ iz başına oylama ─▶ WebSocket olayı
                                                              │
                                                              └─ ilk onay ─▶ HMAC ─▶ veritabanı
```

Tek bir resim algılamadan doğrudan biçim denetimine gider. Akış buna kare atlama, takip ve oylamayı
ekler. Veritabanına yazan yalnızca iki yol vardır: akış ve `anpr watch --store`.

## Modüller (`src/anpr/`)

| Yol | Görevi |
|---|---|
| `detector/` | `Detector` protokolü ve onun fast-alpr uygulaması. Kutuları güven değerleriyle döndürür. |
| `ocr/` | `PlateReader` protokolü ve onun fast-plate-ocr uygulaması. Metni ve karakter başına güven değerlerini döndürür. |
| `tracker/iou.py` | Açgözlü IoU takipçisi. Kararlı iz kimlikleri; 30 kare görülmeyen iz silinir. |
| `postprocess/turkish.py` | 81 il kodu ve plaka düzeni. `34ABC1234` metnini il, harfler ve rakamlar diye ayırır. |
| `postprocess/formats.py` | Her ülke için bir `PlateFormat` (`ANPR_PLATE_FORMATS`) ve `parse_plate`; düzeltilmiş okumalardan önce her biçimde olduğu gibi okumayı dener. |
| `postprocess/confusion.py` | Bir okuma uymazsa harf bloğunun olası üç uzunluğunu dener ve her birine uyması için birbirine benzeyen karakterleri değiştirir. |
| `postprocess/temporal.py` | Bir izin en iyi okumaları üzerinde, güvenle ağırlıklandırılmış, karakter başına çoğunluk oylaması. |
| `postprocess/dewarp.py` | Akışta kırpılmış görüntüyü sabit bir yüksekliğe ölçekler. Gerçek bir perspektif düzeltmesi için yer tutucudur. |
| `pipeline.py` | Tek resim için `infer_image`, akışlar için asenkron `Pipeline`. |
| `video.py` | `anpr watch` için kare kaynakları: video dosyaları, ağ yayınları ve kameralar; canlı kaynaklarda yeniden bağlanma. |
| `watchlist.py` | İzleme listesi; bellekte HMAC özetleri olarak tutulur ve ara sıra veritabanından yeniden yüklenir. |
| `webhook.py` | İzleme listesi eşleşmelerini `ANPR_WEBHOOK_URL` adresine imzalı ve yeniden denenen POST'larla gönderir; aynı plaka için bekleme süresi uygular. |
| `imaging.py` | Yüklenen dosyaları ve kareleri çözer; bildirilen boyutu fazla olanları çözmeden reddeder. |
| `storage/` | `Detection` tablosu (SQLModel), HMAC özetleme, depo (repository) ve saklama süresi görevi. |
| `api/` | FastAPI uygulaması, yolları, web sayfası ve istek kimlikleriyle metrikler için ara katman. |
| `observability.py` | Prometheus metrikleri ve isteğe bağlı OpenTelemetry. |
| `config.py` | pydantic-settings ile `ANPR_` ortam değişkenlerinden okunan ayarlar. |
| `cli.py` | Typer ile yazılmış `anpr` komutu. |

## Tasarım kararları

**Model sınırında protokoller.** İşlem hattı yalnızca `Detector` ve `PlateReader`'ı bilir. Başka bir
dedektör ya da OCR motoru tek bir sınıf yazılarak eklenebilir; testler de sahte modeller kullanır,
böylece çoğu hiçbir model olmadan saniyenin küçük bir kesrinde çalışır.

**Biçim kuralları OCR'ın içinde değil, sonrasında.** OCR modeli geneldir. Türk plakasının
`NN L{1-3} N{2-4}` düzeninde olduğunu bilmek, ayrıştırıcının genel bir modelin yanlış okuduğu
karakterleri, örneğin rakam olması gereken yerdeki bir `O`'yu düzeltmesini sağlar. Her ülkenin kendi
düzenine sahip kendi `PlateFormat`'ı vardır, yani aynı düzeltmeler açık olan her biçimde işler; bkz.
[Plaka biçimleri](/tr/guide/plate-formats/).

**Özet, kaydın tam kıyısında.** Plaka metni bellekte ve yanıtta bulunur; bir satır yazıldığı anda
özetlenir, böylece hiçbir kod yolu onu yanlışlıkla kaydedemez.

**Akışta her şey sınırlı.** Dört karelik bir kuyruk, kuyruk doluyken atılan kareler ve çözmeden önce
boyut denetimi: yavaş bir model ya da kötü niyetli bir istemci servisin sınırsız bellek kullanmasına
yol açamaz.

## 2022 sürümü

Bir YOLOv5 çatalı, EasyOCR ve Flask üzerine kurulan ilk sürüm `legacy/` altında değiştirilmeden
duruyor ve `v0.1.0-legacy` etiketini taşıyor. Bakımı yapılmıyor.
