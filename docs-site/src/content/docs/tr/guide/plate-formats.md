---
title: Plaka biçimleri
description: anpr'ın hangi ülkelerin plakalarını denetleyip düzelttiği ve bunların nasıl seçildiği.
---

Dedektör birçok ülkenin plakasını bulur, ama bir okuma ancak açtığınız plaka düzenlerinden birine
uyarsa kabul edilir. `ANPR_PLATE_FORMATS` bunları tercih sırasıyla listeler. Varsayılan `tr`,
yani yalnızca Türkiye. 0.3.0'dan beri.

```bash
export ANPR_PLATE_FORMATS=de,fr,tr
```

## Ülkeler

| Kod | Ülke | Düzen | Standart yazım | Bölge |
|---|---|---|---|---|
| `tr` | Türkiye | 2 haneli il kodu, 1–3 harf, 2–4 rakam | `34 ABC 1234` | il kodu ve adı |
| `de` | Almanya | 2–5 harf, sıfırla başlamayan 1–4 rakam, toplam en çok 8, isteğe bağlı `E` ya da `H` | `MAB 1234` | yok |
| `fr` | Fransa (SIV, 2009'dan beri) | 2 harf, 3 rakam, 2 harf; I, O, U yok; `SS` ve `000` verilmez | `AB-123-CD` | yok |
| `it` | İtalya (1994'ten beri) | 2 harf, 3 rakam, 2 harf; I, O, Q, U yok | `AB 123 CD` | yok |
| `es` | İspanya (2000'den beri) | 4 rakam, 3 sessiz harf; Ñ ve Q yok | `1234 BCD` | yok |
| `be` | Belçika (2010'dan beri) | 1 rakam, 3 harf, 3 rakam | `1-ABC-123` | yok |
| `pt` | Portekiz | `AA-00-AA`, `00-AA-00`, `00-00-AA` ya da `AA-00-00` | `AA-00-AA` | yok |

Alman plakaları `M-AB 1234` olarak değil, harfler ve rakamlar olarak, `MAB 1234` diye çıkar. Bölge
kodu bir ile üç harf arasında olduğu için bölge kodu ile geri kalanın nerede ayrıldığı yalnızca
karakterlere bakarak anlaşılamaz; bunun için bölge kodlarının listesi gerekir ve o liste henüz
yerleşik değil. Eşleştirme bundan etkilenmez: özetler boşluk ve tireyi yok sayar, yani
[izleme listesine](/tr/guide/watchlist/) `M-AB 1234` diye eklenen plaka yine eşleşir.

Yalnızca Türk plakalarında bölge vardır. Diğerlerinde `province_code` ve `province_name` her
yanıtta `null` olur.

## Bir okuma nasıl denetlenir

1. OCR metnindeki boşluklar atılır ve büyük harfe çevrilir.
2. `ANPR_PLATE_FORMATS` içindeki her biçim metni olduğu gibi, sırayla dener. Kabul eden ilk biçim
   kazanır.
3. Hiçbiri kabul etmezse her biçim, birbirine benzeyen karakterleri düzeninin o konumda beklediği
   harf ya da rakamla değiştirerek yeniden dener: `0` ile `O`, `1` ile `I`, `8` ile `B`, `5` ile
   `S`, `2` ile `Z`, `6` ile `G`. İspanyol plakası `I2B4BCD`, `1284BCD` olur.
4. Bu da olmazsa `parsed` değeri `null` olur ve okuma akışta onaylanmaz.

Herhangi bir biçimde olduğu gibi geçerli olan okuma, düzeltilmiş okumaya üstün gelir. Aynı metni
iki biçim de olduğu gibi kabul ederse listede önce gelen kazanır: `AB123CD` hem Fransa'da hem
İtalya'da geçerlidir, `ANPR_PLATE_FORMATS=it,fr` ile İtalyan plakası olarak okunur.

Yalnızca beklediğiniz ülkeleri açın. Her fazla biçim, yanlış bir okumanın geçerli görünmesi için bir
yol daha demektir; özellikle kısa Portekiz düzenleri altı karakterli pek çok metni kabul eder.

## Yanıtlarda

Her `parsed` nesnesinde tablodaki kodu taşıyan bir `country` alanı bulunur:

```json
"parsed": {
  "country": "de",
  "province_code": null,
  "province_name": null,
  "letters": "MAB",
  "digits": "1234",
  "canonical": "MAB 1234"
}
```

`anpr infer` ve `anpr watch` Türk plakalarında ili, diğerlerinde ülke kodunu gösterir. Kaydedilen
algılamaların bir `country` sütunu vardır ve `GET /api/v1/detections` bunu döndürür. Daha eski bir
sürümün oluşturduğu veritabanına servis bu sütunu açılışta ekler; o zamandan önceki satırlarda
değer `null` kalır. `anpr_confirmed_plates_total` metriğinin bir `country` etiketi vardır.
