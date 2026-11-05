---
title: Katkıda bulunma
description: Projeyi kurun, CI'ın çalıştırdığı denetimleri çalıştırın ve bir pull request açın.
---

Hata bildirimleri, düzeltmeler, yeni özellikler ve belgeler, hepsi memnuniyetle karşılanır. Küçük
bir düzeltmeden büyük her iş için önce bir issue açın; böylece kodu yazmadan önce kapsam üzerinde
anlaşılır.

## Kurulum

[uv](https://docs.astral.sh/uv/) ve Python 3.11 ya da daha yenisi gerekir.

```bash
git clone https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR.git
cd Automatic_Number_Plate_Recognition_YOLO_OCR
uv sync --all-extras --group dev
uv run pre-commit install   # isteğe bağlı: denetimleri her commit'te çalıştırır
```

## Denetimler

CI bunları Python 3.11, 3.12, 3.13 ve 3.14 üzerinde çalıştırır, sonra Docker imajını derler ve
`/health` adresini çağırır. Göndermeden önce çalıştırın:

```bash
make lint   # ruff check ve ruff format --check
make type   # pyright
make test   # pytest
```

Varsayılan test takımı sahte modeller kullanır ve ağ gerektirmez. Gerçek modellerle uçtan uca test
modelleri ilk çalıştırmada indirir (yaklaşık 30 MB):

```bash
ANPR_RUN_REAL=1 uv run pytest tests/integration/test_pipeline_real.py
```

## Kurallar

- Dal adları `feat/`, `fix/`, `docs/`, `chore/`, `ci/`, `test/` ya da `refactor/` ile başlar.
- Commit mesajları [Conventional Commits](https://www.conventionalcommits.org/) kuralına uyar:
  `feat:`, `fix:`, `docs:` vb.
- Her pull request tek bir mantıksal değişiklik içerir.
- Kod `ruff format` ile 100 sütunda biçimlenir ve pyright ile tip denetiminden geçer. Herkese açık
  fonksiyonların tip ipuçları eksiksizdir.
- Plaka metni asla kaydedilmez. Bir plakayı kalıcı hale getiren her şey
  `storage.hashing.hash_plate` üzerinden geçer.

## Belgeler

Bu site `docs-site/` içinde yaşar ve [Astro Starlight](https://starlight.astro.build) ile derlenir.
Sayfalar `docs-site/src/content/docs/` altındaki Markdown dosyalarıdır; Türkçe çeviriler `tr/`
altındadır.

```bash
cd docs-site
npm install
npm run dev   # http://localhost:4321
```

`main` dalına `docs-site/` içinde değişiklik yapan her push siteyi yayımlar.
