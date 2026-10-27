---
title: İlk plakanız
description: Komut satırından bir resimdeki plakayı okuyun, sonra servisi başlatıp web sayfasını deneyin.
---

## Komut satırından

`anpr infer` komutuna içinde araba olan bir fotoğraf verin:

```bash
anpr infer araba.jpg
```

Dedektörün bulduğu her plaka için bir satırlık tablo çıkar:

| Sütun | Anlamı |
|---|---|
| BBox | Plakanın resimdeki yeri, piksel cinsinden `(x1,y1)-(x2,y2)`. |
| Det conf | Dedektörün bunun plaka olduğundan ne kadar emin olduğu, 0 ile 1 arası. |
| Raw OCR | OCR modelinin okuduğu karakterler, olduğu gibi. |
| Plate | Türk plaka biçimi denetiminden sonraki okuma, örneğin `34 ABC 1234`. Biçime uymuyorsa `(invalid)`. |
| Province | Kodun ait olduğu il, örneğin `İstanbul`. |

Aynı sonucu betiklerde kullanmak üzere JSON olarak almak için `--json` ekleyin:

```bash
anpr infer araba.jpg --json
```

Komut satırı aracı veritabanına hiçbir şey yazmaz.

## Servis ve web sayfası

Servisi başlatın:

```bash
anpr serve
```

<http://localhost:8000> adresini açın. Sayfada iki yol var:

- **Resim yükleyin.** Resim [`POST /api/v1/infer`](/tr/guide/http-api/) adresine gider, plakalar
  altında listelenir.
- **Web kamerasını açın.** Sayfa kareleri [`/ws/stream`](/tr/guide/websocket/) üzerinden gönderir ve
  her plakayı okundukça gösterir. Bir plaka yeterince karede okununca onaylandı olarak işaretlenir
  ve özet olarak kaydedilir.

FastAPI'nin ürettiği etkileşimli API belgesi <http://localhost:8000/docs> adresinde.

`ANPR_PLATE_HMAC_PEPPER` ayarlamadıysanız servis rastgele bir pepper üretir ve uyarı verir. Denemek
için sorun değil. Gerçek kullanımda bir tane ayarlayın; bkz. [Yapılandırma](/tr/guide/configuration/).
