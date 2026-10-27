---
title: İzleme listesi ve webhook
description: anpr'a aranacak plakaları verin, biri onaylandığında bir HTTP çağrısı alın.
---

İzleme listesi, haberdar olmak istediğiniz plakalardır. [WebSocket akışı](/tr/guide/websocket/) ya
da [`anpr watch`](/tr/guide/cli/#anpr-watch-kaynak) listedeki bir plakayı onayladığında anpr,
belirlediğiniz adrese JSON bir mesaj gönderir (`POST`). 0.3.0'dan beri.

Plakalar gelir gelmez, kaydedilen algılamalarla aynı pepper'la özetlenir. Yani izleme listesinde
de okunabilir plaka yoktur. Bkz. [Gizlilik](/tr/operations/privacy/).

## Kurulum

1. Pepper'ı sabitleyin. Listedeki her kayıt bir özettir ve rastgele pepper her yeniden başlatmada
   özeti değiştirir. `ANPR_PLATE_HMAC_PEPPER` ayarlı değilken anpr listeye plaka eklemez.

   ```bash
   export ANPR_PLATE_HMAC_PEPPER=$(anpr generate-pepper)
   ```

2. Eşleşmelerin nereye gideceğini ve imzalanacakları gizli değeri verin:

   ```bash
   export ANPR_WEBHOOK_URL=https://ornek.com/hooks/anpr
   export ANPR_WEBHOOK_SECRET=$(anpr generate-api-key)
   ```

3. Plakaları ekleyin:

   ```bash
   anpr watchlist add "34 ABC 1234" --label "dağıtım minibüsü"
   anpr watchlist import plakalar.txt
   ```

Boşluk, tire ve büyük-küçük harf fark etmez: `34-abc-1234` ile `34 ABC 1234` aynı kayıttır. Çalışan
sunucu listeyi 30 saniyede bir veritabanından yeniden okur. Komut satırından eklenen plakalar bu
sayede yeniden başlatmaya gerek kalmadan sunucuya ulaşır.

Etiket her uyarıyla birlikte gönderilir, alan taraf neye baktığını bilsin diye. Etikete bir rol ya
da araç yazın ("dağıtım minibüsü", "2 numaralı kapı yüklenicisi"). Oraya yazılan bir kişi adı açık
metin halinde kişisel veri olur.

## Komutlar

| Komut | Ne yapar |
|---|---|
| `anpr watchlist add PLAKA [--label METİN]` | Plaka ekler, listede varsa etiketini değiştirir. `--label` olmadan yeniden eklemek eski etiketi korur. |
| `anpr watchlist import DOSYA` | Bir metin dosyasındaki bütün plakaları ekler: her satıra bir plaka, isterseniz `plaka,etiket`. Boş satırlar ve `#` ile başlayan satırlar atlanır. |
| `anpr watchlist remove PLAKA` | Plakayı çıkarır. Plaka metnini ya da 64 karakterlik özetini alır. Plaka listede değilse 1 koduyla çıkar. |
| `anpr watchlist list [--json]` | Özetleri, etiketleri ve her birinin ne zaman eklendiğini gösterir. |

```text
# plakalar.txt
34 ABC 1234, dağıtım minibüsü
06 AA 42
```

## HTTP uçları

Aynı liste HTTP üzerinden. Bu uçlar yalnızca `ANPR_API_KEY` ayarlıyken çalışır ve her çağrıda
[anahtarı](/tr/guide/http-api/#api-anahtarı) ister; anahtar ayarlı değilse `403` döner. Plaka
eklemek ve listeyi görmek plakanın özetini gösterir. Açık bir API'de herkes bunu kullanarak
istediği bir plakanın `/api/v1/detections` içinde olup olmadığına bakabilirdi.

```bash
curl -X POST http://localhost:8000/api/v1/watchlist \
  -H "Content-Type: application/json" \
  -d '{"plate": "34 ABC 1234", "label": "dağıtım minibüsü"}'
```

```json
{
  "plate_hmac": "9f2c…e41a",
  "label": "dağıtım minibüsü",
  "created_at": "2026-10-27T14:02:51.120000"
}
```

| Uç | Yanıt |
|---|---|
| `GET /api/v1/watchlist` | Bütün kayıtlar, en eskisi önce. |
| `POST /api/v1/watchlist` | `201` ve yeni kayıt. Gövde: `plate` (zorunlu, en çok 32 karakter) ve `label` (isteğe bağlı, en çok 200). Listedeki bir plakayı yeniden göndermek etiketini günceller; `label` verilmezse eski etiket kalır. Pepper ayarlı değilken `409`, `plate` içinde harf ya da rakam yoksa `422`. |
| `DELETE /api/v1/watchlist/{plate_hmac}` | `204`, özet listede yoksa `404`. |

## Webhook çağrısı

Eşleşmede anpr JSON gövdeli tek bir `POST` gönderir:

```json
{
  "event": "watchlist.match",
  "timestamp": "2026-10-27T14:05:09.731204+00:00",
  "source": "stream",
  "plate_hmac": "9f2c…e41a",
  "label": "dağıtım minibüsü",
  "province_code": "34",
  "track_id": 7,
  "frame_idx": 118,
  "confidence": 0.88
}
```

`source`, `/ws/stream` için `stream`, `anpr watch` için `watch` olur. Gövdede plaka metni yoktur;
hangi kaydın eşleştiğini etiket ve özet söyler.

Çağrı arka planda yapılır, yavaş yanıt veren bir alıcı videoyu bekletmez. anpr yanıtı en çok
`ANPR_WEBHOOK_TIMEOUT_SECONDS` saniye bekler ve her `2xx` yanıtı teslim edilmiş sayar. Ağ hatası,
`429` ya da `5xx` alırsa önce bir, sonra iki saniye bekleyip yeniden dener, toplam üç deneme.
Yönlendirmeler (`3xx`) ve diğer `4xx` yanıtları başarısız sayılır, yeniden denenmez. Yönlendirme
izlenmez; `ANPR_WEBHOOK_URL` değerine son adresi, `https://` ile birlikte yazın. Bütün denemeler
başarısız olursa bekleme süresi kaldırılır ve plaka bir sonraki görülüşünde yeniden denenir. Aynı plaka en çok `ANPR_WEBHOOK_COOLDOWN_SECONDS` saniyede bir
(varsayılan 60) bildirilir. Durup yeniden hareket eden bir araç çoğu zaman yeni bir iz olarak geri
gelir.

## İmzayı doğrulamak

`ANPR_WEBHOOK_SECRET` ayarlıysa her çağrıda iki başlık bulunur:

```text
X-ANPR-Timestamp: 1793109909
X-ANPR-Signature: sha256=5d0c…
```

İmza; zaman damgası, bir nokta ve ham gövdenin, gizli değerle anahtarlanmış HMAC-SHA256 özetidir
(onaltılık). Mesaja güvenmeden önce imzayı denetleyin. Kaydedilmiş bir çağrının yeniden
oynatılamaması için eski zaman damgalarını da reddedin:

```python
import hashlib
import hmac
import time


def dogrula(gizli: str, basliklar: dict[str, str], govde: bytes, en_eski: int = 300) -> bool:
    zaman = basliklar["X-ANPR-Timestamp"]
    if abs(time.time() - int(zaman)) > en_eski:
        return False
    mac = hmac.new(gizli.encode(), zaman.encode() + b"." + govde, hashlib.sha256)
    return hmac.compare_digest(f"sha256={mac.hexdigest()}", basliklar["X-ANPR-Signature"])
```

Gövdeyi geldiği gibi kullanın; JSON'u ayrıştırıp yeniden yazmak baytları değiştirir.

## Nerede çalışır

- **`/ws/stream`** onaylanan her plakayı denetler.
- **`anpr watch`**, `ANPR_WEBHOOK_URL` ayarlıyken ya da `--store` açıkken listeyi denetler; veritabanını
  yalnızca bu durumlarda açar. Eşleşmeler çıktıda `watchlist` diye işaretlenir, JSON satırlarına
  bir `watchlist` alanı eklenir.
- **`POST /api/v1/infer`** denetlemez: tek bir resimde onaylanmış okuma olmaz.

`anpr_watchlist_matches_total` ve `anpr_webhook_deliveries_total` metrikleri eşleşmeleri ve teslim
sonuçlarını sayar. Bkz. [İzleme](/tr/operations/monitoring/).
