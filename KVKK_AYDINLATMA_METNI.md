# KVKK Aydınlatma Metni — Plaka Tanıma Sistemi

> **Bu belge bir başlangıç şablonudur, hukuki tavsiye değildir.** Sisteminizi kurmadan
> önce kendi veri sorumlusu/işleyen ilişkinizi, hukuki sebebinizi, saklama sürelerinizi,
> teknik ve idari tedbirlerinizi bir hukuk danışmanı ile gözden geçirin. Ticari/kamu
> dağıtımlar için VERBİS kaydı, açık rıza akışı, DPIA (etki değerlendirmesi) ve
> sözleşme zinciri (varsa veri işleyen, alt işleyenler) ayrıca değerlendirilmelidir.

6698 sayılı **Kişisel Verilerin Korunması Kanunu** ("KVKK") uyarınca; bu belge,
araç plakanızın bu sistem tarafından işlenmesi sürecine ilişkin sizi
aydınlatmak amacıyla hazırlanmıştır.

## 1. Veri Sorumlusu

- **Veri Sorumlusu:** _[Kurum/Şirket adı]_
- **Adres:** _[Tam adres]_
- **VKN/MERSİS:** _[Numara]_
- **İletişim:** _[E-posta / telefon]_
- **VERBİS:** _[Sicil numarası, varsa]_

## 2. İşlenen Kişisel Veriler

Bu sistem aracılığıyla aşağıdaki kişisel veriler işlenmektedir:

| Veri Kategorisi | Veri Türü | Toplama Yöntemi |
|---|---|---|
| Araç ve Bağlı Kişi Verisi | Plaka metni (HMAC-SHA256 ile **şifrelenmiş** olarak saklanır; ham metin **kalıcı şekilde saklanmaz**) | Kameradan/yüklenen görselden otomatik tespit + OCR |
| İşlem Verisi | Tespit zaman damgası, takip kimliği, güven skoru, görüntüdeki sınır kutusu koordinatları | Otomatik üretim |
| _(Opsiyonel)_ Görsel | Kamera görüntüsünde yer alan diğer kişi/araç/ortam verisi | Kameradan/yüklemeden |

> ⚠ **Önemli:** Sistem varsayılan olarak plaka metnini saklamaz; yalnızca
> deployment-spesifik bir gizli anahtar (`ANPR_PLATE_HMAC_PEPPER`) ile
> HMAC-SHA256 özetini saklar. Bu özet, **aynı plaka için aynı değeri**
> üretmesine rağmen anahtar olmadan tersine çevrilemez.

## 3. Kişisel Verilerin İşlenme Amaçları

- _[Örnek] Tesise/Otoparka giriş-çıkış kontrolünün otomatik gerçekleştirilmesi_
- _[Örnek] İzinsiz araç girişi durumunda güvenlik birimini uyarma_
- _[Örnek] Otopark süresi bazlı ücretlendirme_
- Yasal yükümlülüklerin yerine getirilmesi
- Sistemin teknik bakım, hata ayıklama ve iyileştirilmesi (toplu / anonim metriklerle)

## 4. Kişisel Verilerin İşlenmesinin Hukuki Sebebi

Aşağıdakilerden uygun olan(lar)ı seçilmelidir (KVKK md. 5):

- [ ] Açık rıza (md. 5/1)
- [ ] Kanunlarda açıkça öngörülmesi (md. 5/2-a)
- [ ] Bir sözleşmenin kurulması veya ifasıyla doğrudan doğruya ilgili olması (md. 5/2-c)
- [ ] Veri sorumlusunun hukuki yükümlülüğünü yerine getirebilmesi için zorunlu olması (md. 5/2-ç)
- [ ] **İlgili kişinin temel hak ve özgürlüklerine zarar vermemek kaydıyla, veri sorumlusunun meşru menfaatleri için zorunlu olması** (md. 5/2-f) — özel güvenlik / otopark erişim kontrolü senaryosunda yaygın olarak kullanılır

## 5. Kişisel Verilerin Aktarımı

- **Yurtiçi aktarım:** _[Hangi alıcı gruplarına aktarılıyor? Örn. yetkili kamu kurumları, hizmet aldığınız teknoloji sağlayıcılar]_
- **Yurtdışı aktarım:** _[Varsa, hangi ülke + KVKK Kurulu yeterlilik kararı / taahhütname / BCR]_

## 6. Saklama Süresi

Bu sistemde varsayılan saklama süresi **`ANPR_RETENTION_HOURS` ortam değişkeni**
ile yapılandırılır (kurulum varsayılanı 720 saat = 30 gün). Süre dolduğunda
arka plan iş parçacığı kayıtları otomatik olarak siler.

- _[Sizin saklama süreniz, ör. "Otopark giriş-çıkış tutanakları 6 ay süreyle saklanır."]_
- Saklama ve İmha Politikası: _[varsa, link / referans]_

## 7. İlgili Kişinin Hakları (KVKK md. 11)

Kişisel verisi işlenen kişi olarak siz, aşağıdaki haklara sahipsiniz:

- Kişisel verinizin işlenip işlenmediğini öğrenme
- İşlenmişse buna ilişkin bilgi talep etme
- İşlenme amacını ve amacına uygun kullanılıp kullanılmadığını öğrenme
- Yurt içinde / yurt dışında aktarıldığı üçüncü kişileri bilme
- Eksik veya yanlış işlenmiş ise düzeltilmesini isteme
- KVKK ve ilgili mevzuat çerçevesinde silinmesini / yok edilmesini isteme
- Düzeltme / silme / yok etme işlemlerinin üçüncü kişilere bildirilmesini isteme
- Otomatik sistemlerle analiz edilmesi sonucu aleyhinize sonuç doğmasına itiraz etme
- Kanuna aykırı işleme nedeniyle zarara uğradığınızda zararın giderilmesini talep etme

Başvurularınızı, **Veri Sorumlusuna Başvuru Usul ve Esasları Hakkında Tebliğ**
hükümlerine uygun olarak aşağıdaki kanalları kullanarak iletebilirsiniz:

- E-posta: _[KEP veya kayıtlı e-posta]_
- Yazılı: _[Adres]_
- Form: _[Varsa link]_

Başvurunuz en geç **30 gün** içinde yanıtlanır. Ücretsizdir; ayrıca bir maliyet
gerekirse Kurul'ca belirlenen tarifeye göre ücret talep edilebilir.

## 8. Teknik ve İdari Tedbirler

Bu sistemde alınan başlıca tedbirler:

- **Şifreleme:** Plaka metni HMAC-SHA256 + deployment-spesifik gizli anahtar
  (pepper) ile özetlenip saklanır. Ham metin veritabanına yazılmaz.
- **Erişim kontrolü:** API ve veritabanı yalnızca yetkili kişiler tarafından
  erişilebilir; CORS / auth katmanları kurulum sırasında yapılandırılmalıdır.
- **Asgari veri:** Yalnızca amaç için gerekli alanlar saklanır (zaman damgası,
  güven skoru, sınır kutusu, plaka özeti, il kodu).
- **Otomatik silme:** Saklama süresi aşıldığında arka plan iş parçacığı
  kayıtları siler.
- **Loglama:** Erişim ve sistem olayları yapılandırılmış formatta loglanır
  (structlog + opsiyonel OpenTelemetry).
- **Süreç:** _[İlgili politikalar — Bilgi Güvenliği Politikası, İmha Politikası, vb.]_

---

_Aydınlatma Metni Versiyon:_ 1.0  
_Yürürlük Tarihi:_ _[YYYY-AA-GG]_  
_Son Güncelleme:_ _[YYYY-AA-GG]_
