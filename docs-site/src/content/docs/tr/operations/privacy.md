---
title: Gizlilik, KVKK ve GDPR
description: anpr-pipeline neden plaka yerine anahtarlı bir özet saklar, eşleştirme nasıl yine de çalışır ve işletmeci olarak size ne kalır.
---

Plaka bir aracı, onun üzerinden de bir kişiyi tanımlar. Hem KVKK'ya (6698 sayılı Kanun, madde
3(d)) hem AB'nin GDPR'ına (2016/679 sayılı Tüzük, madde 4(1)) göre kişisel veridir. anpr-pipeline,
veritabanında hiçbir plakanın okunabilir hâlde durmaması için tasarlandı.

## Ne kaydedilir

[WebSocket akışı](/tr/guide/websocket/) bir plakayı onayladığında tek bir satır yazılır:

| Sütun | Örnek |
|---|---|
| `plate_hmac` | `9f2c…e41a`, plakanın HMAC-SHA256 özeti |
| `province_code` | `34` |
| `confidence` | `0.88` |
| kutu | `412, 388, 618, 436` |
| `track_id`, `timestamp`, `confirmed` | |

`anpr watch --store` da aynı satırı yazar. Plaka metni, resim ve kare kaydedilmez.
`POST /api/v1/infer` ve `anpr infer` hiçbir şey kaydetmez.

[İzleme listesi](/tr/guide/watchlist/) de aynı şekilde tutulur: eklediğiniz her plaka geldiği anda
özetlenir, yalnızca özet ve etiketiniz yazılır. Webhook çağrıları plakayı değil, özeti ve etiketi
taşır.

## Özet nasıl çalışır

Plaka önce tek bir biçime getirilir (yalnızca harfler ve rakamlar kalır, büyük harfe çevrilir);
böylece `34 abc 1234`, `34-ABC-1234` ve `34ABC1234` aynı özeti verir. Sonra `ANPR_PLATE_HMAC_PEPPER` anahtarıyla HMAC-SHA256 özeti
alınır.

- **Eşleştirme yine çalışır.** Aynı plaka, aynı pepper'la her zaman aynı özeti verir. Bu sayede
  ziyaretleri sayabilir ya da bildiğiniz bir plakayı arayabilirsiniz.
- **Tahmin işe yaramaz.** Türk plakaları bilinen bir kalıba uyduğu için düz bir özet, bütün
  plakalar denenerek geri çözülebilirdi. Pepper'ı bilmeyen biri karşılaştıracağı özetleri
  hesaplayamaz.
- **Pepper'ı güvende ve sabit tutun.** Bir parola gibi, veritabanının dışında saklayın.
  Değiştirirseniz aynı plakanın eski ve yeni özetleri artık eşleşmez.

Bilinen bir plakayı kayıtlı satırlarda aramak için:

```python
from anpr.storage.hashing import hash_plate

aranan = hash_plate("34 ABC 1234", pepper)
# `aranan` değerini GET /api/v1/detections yanıtındaki plate_hmac ile karşılaştırın
```

## Saklama süresi

Arka planda çalışan bir görev, `ANPR_RETENTION_HOURS` değerinden eski satırları siler. Varsayılan
720 saat, yani 30 gün. Görev servis açılırken ve sonra her saat çalışır. Süreyi amacınızın gerçekten
gerektirdiği kadar tutun.

## Size kalanlar

Yazılım, kaydedilen veriyi en aza indirir. Bir kurulumu tek başına hukuka uygun hâle getirmez. Veri
sorumlusu olarak başka şeylerin yanında şunlar size düşer:

- plaka kaydetmek için bir hukuki sebep ve belirli bir amaç,
- kaydedilen kişiler için bir aydınlatma metni. Depoda Türkçe bir başlangıç taslağı var:
  [`KVKK_AYDINLATMA_METNI.md`](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/blob/main/KVKK_AYDINLATMA_METNI.md).
  Kendi durumunuza uyarlayın; hukuki tavsiye değildir,
- servise ve veritabanına erişim denetimi: `ANPR_API_KEY` ayarlayın ve HTTPS üzerinden sunun,
- daraltılmış bir `ANPR_CORS_ORIGINS`.

Web sayfası, siz izlerken okunabilir plakayı tarayıcıda gösterir ama saklanmak üzere hiçbir yere
göndermez.
