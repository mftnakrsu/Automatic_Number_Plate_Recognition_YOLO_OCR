---
title: Yol haritası
description: anpr-pipeline'ın henüz yapamadıkları ve sırada olanlar.
---

## Bilinen sınırlar

Şu anki sürümdeki (0.3.x) gerçek eksikler bunlar.

- **Tek, ortak bir API anahtarı.** `ANPR_API_KEY` bütün istemciler için tek bir anahtardır;
  kullanıcı, rol ya da istemci başına anahtar yoktur.
- **Yedi ülke.** Okumalar Türkiye, Almanya, Fransa, İtalya, İspanya, Belçika ve Portekiz düzenlerine
  göre denetlenir ([Plaka biçimleri](/tr/guide/plate-formats/)). Diğer plakalar `parsed: null`
  olarak döner; Alman plakaları da bölge kodu ve geri kalan diye ayrılmaz.
- **Perspektif düzeltmesi yok.** Dik bir açıdan görülen plakalar OCR'dan önce düzleştirilmez; akış
  kırpılmış görüntüyü yalnızca sabit bir yüksekliğe ölçekler.
- **İki ayar henüz hiçbir şey yapmıyor.** `ANPR_SR_MIN_PLATE_WIDTH` (küçük plakalar için süper
  çözünürlük) ve `ANPR_OCR_FALLBACK_THRESHOLD` (emin olunmayan okumalar için ikinci bir OCR motoru)
  yalnızca yer tutucudur.

## Sırada

Kabaca bu sırayla:

1. **Daha çok ülke**: önce Hollanda, Polonya ve Avusturya; bir de `MAB 1234` plakasının
   `M-AB 1234` olarak gösterilebilmesi için Alman bölge kodlarının listesi.

Fikirler ve pull request'ler
[GitHub'da](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/issues)
memnuniyetle karşılanır.
