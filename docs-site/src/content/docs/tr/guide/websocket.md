---
title: WebSocket akışı
description: /ws/stream adresine video kareleri gönderin, plaka okumalarını alın; takip ve kareler arası onay dahil.
---

`/ws/stream` video içindir: web kamerası, bir kamera yayını ya da bir dosyadan okunan kareler. Aynı
makinedeki bir dosyayı ya da RTSP kamerayı okumak için [`anpr watch`](/tr/guide/cli/#anpr-watch-kaynak)
aynı işi sunucusuz yapar. Siz
kareleri gönderirsiniz, servis okuduklarını geri yollar. Her plakayı kareden kareye izler ve
yeterince okununca onaylar.

## Protokol

1. `ws://localhost:8000/ws/stream` adresine bir WebSocket bağlantısı açın.
2. Her kareyi, kodlanmış bir resim (genellikle JPEG) içeren tek bir **ikili (binary)** mesaj olarak
   gönderin.
3. **Metin** mesajları alın. Her biri, işlenen bir karedeki bir plaka için bir JSON nesnesidir.

Her bağlantının kendi takipçisi ve oylaması vardır, iki akışın plakaları birbirine karışmaz.

### API anahtarıyla

`ANPR_API_KEY` ayarlıysa el sıkışma anahtar ister, yoksa bağlantı reddedilir (kapanış kodu `1008`).
Başlık ekleyebilen bir istemci `X-API-Key` gönderir. Tarayıcılar WebSocket'e başlık ekleyemez; bu
yüzden anahtarı bir alt protokol olarak sunarlar, servis de `anpr` ile yanıt verir:

```js
new WebSocket('wss://anpr.example.com/ws/stream', ['anpr', `anpr.key.${key}`]);
```

Böylece anahtar, erişim günlüklerine yazılacağı URL'ye hiç girmez.

## Ne geri gelir

```json
{
  "frame_idx": 42,
  "track_id": 3,
  "bbox": [412, 388, 618, 436],
  "detection_confidence": 0.91,
  "raw_text": "34ABC1234",
  "normalized_text": "34ABC1234",
  "parsed": {
    "canonical": "34 ABC 1234",
    "country": "tr",
    "province_code": "34",
    "province_name": "İstanbul"
  },
  "confirmed": "34 ABC 1234"
}
```

| Alan | Anlamı |
|---|---|
| `frame_idx` | Bu bağlantıdaki karenin sırası, 0'dan başlayarak. |
| `track_id` | Aynı plaka görüntüde kaldıkça değişmez. |
| `confirmed` | Bu iz için oylamaya yetecek okuma birikene kadar `null`, sonra üzerinde anlaşılan plaka. |

Öteki alanlar [`POST /api/v1/infer`](/tr/guide/http-api/) ile aynı.

## Bir plaka nasıl onaylanır

- **Kare atlama.** Yalnızca her N. kare işlenir (`ANPR_DETECT_EVERY_N_FRAMES`, varsayılan 3). Öteki
  kareler çözülür ama modellere hiç gitmez. Bu, canlı yayını ucuz tutar.
- **Takip.** İşlenen her karede her kutu, son kutusuyla en çok örtüşen ize bağlanır; örtüşme en az
  %30 olmalı (IoU 0,3). Hiçbir ize uymayan kutu yeni bir iz başlatır. İşlenen 30 kare boyunca
  görülmeyen iz silinir.
- **Oylama.** Bir izin her geçerli okuması, son 8 okumayı tutan bir geçmişe girer. İz
  `ANPR_MIN_TRACK_DWELL` kadar işlenmiş karede okununca (varsayılan 5), en emin üç okuma karakter
  karakter oy verir; her oy kendi güven skoruyla ağırlıklandırılır. Kazanan, onaylanan plakadır.
- **Kayıt.** Bir iz ilk kez onaylandığında tek bir satır yazılır: plakanın özeti, il kodu, güven
  skoru ve kutusu. Aynı izin sonraki mesajları yeniden kaydedilmez.

## Geri basınç ve sınırlar

Servis en fazla dört kareyi bekletir. İşleyebileceğinden hızlı gönderirseniz yeni kareler sıraya
alınmaz, atılır; böylece gecikme kısa kalır.

`ANPR_MAX_WS_FRAME_BYTES` değerinden (varsayılan 4 MB) büyük ya da `ANPR_MAX_IMAGE_PIXELS` değerinden
fazla piksel içeren bir kare, bağlantıyı `1009` koduyla kapatır. Çözülemeyen bir kare atlanır ve akış
sürer.

## Örnek istemci

Bir video dosyasını akıtan Python istemcisi; `opencv-python` ve `websockets` kullanır:

```python
import asyncio, json
import cv2, websockets

async def main() -> None:
    cap = cv2.VideoCapture("trafik.mp4")
    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:

        async def receive() -> None:
            async for message in ws:
                event = json.loads(message)
                if event["confirmed"]:
                    print(event["track_id"], event["confirmed"])

        listener = asyncio.create_task(receive())
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, jpeg = cv2.imencode(".jpg", frame)
            await ws.send(jpeg.tobytes())
            await asyncio.sleep(1 / 25)  # aşağı yukarı videonun kare hızı
        await asyncio.sleep(2)  # son okumaların gelmesini bekle
        listener.cancel()

asyncio.run(main())
```

API anahtarı varsa `websockets.connect` çağrısına `additional_headers={"X-API-Key": anahtar}`
ekleyin.

`/` adresindeki web sayfası aynı işi tarayıcıda web kameranızla yapar. Servis anahtar istiyorsa
sayfa bunun için bir alan gösterir ve anahtarı yalnızca o tarayıcı sekmesinde tutar.
