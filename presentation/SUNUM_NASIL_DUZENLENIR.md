# Sunumu nasıl düzenlersiniz?

Bu klasörde **iki format** var; ikisini de özgürce değiştirebilirsiniz.

## 1. Markdown (Marp) — `QOS_Sunum.md`

**En kolay düzenleme:** Her slayt `---` ile ayrılır. Metin, tablolar ve kod bloklarını doğrudan değiştirin.

### VS Code / Cursor

1. [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode) eklentisini kurun.
2. `QOS_Sunum.md` dosyasını açın.
3. Sağ üstten **Open Preview** veya komut paleti: `Marp: Export slide deck`.

**Dışa aktarma:** PDF veya **PowerPoint (.pptx)** — sunumdan sonra PowerPoint’te ince ayar yapabilirsiniz.

### Komut satırı (isteğe bağlı)

```bash
npm install -g @marp-team/marp-cli
marp presentation/QOS_Sunum.md --pptx -o presentation/QOS_Sunum.pptx
marp presentation/QOS_Sunum.md --pdf  -o presentation/QOS_Sunum.pdf
```

## 2. PowerPoint — `QOS_Sunum.pptx`

`generate_pptx.py` ile üretilir veya hazır dosyayı doğrudan PowerPoint / LibreOffice Impress ile açın.

```bash
python presentation/generate_pptx.py
```

Her slayt düzenlenebilir; renkler ve görselleri kendi üniversite şablonunuza göre uyarlayın.

## Sunum günü — canlı demo

```bash
python run_full_demo.py --video input.mp4 --model model_sharp_ep10_fixed.pth --profile medium
```

Tarayıcı sekmeleri:

- http://localhost:8080
- http://localhost:8080/pipeline.html
- http://localhost:8080/comparison.html (önce `make_web_frames.py` gerekebilir)

## Kişiselleştirmeniz gereken yerler

- Kapak: ad, bölüm, tarih
- “Gelecek çalışmalar” ve “Sınırlamalar” slaytları
- PSNR/SSIM tablosu (kendi `metrics_report.txt` değerleriniz)
- Referanslar slaytı
