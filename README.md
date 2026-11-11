

https://github.com/user-attachments/assets/61fae235-2f97-4e08-8f9f-35b99bad7062

# EnerjiRadar

**Saatlik elektrik tüketimi tahmini, tahmin aralığı kalibrasyonu ve "beklenenden fazla tüketim" tespiti.**

Üç yöntemi (geçen haftanın aynı saati, gradyan artırma, PyTorch LSTM) aynı veri bölmesi, aynı ölçütlerle karşılaştırır. Her tahminin %90 aralığını üretir, aralığın gerçekten %90'ı kapsayıp kapsamadığını ölçer, gerçek tüketimin aralığın üstüne çıktığı saatleri SQL ile analiz eder. Yeni bir algoritma icat edildiği iddiası yoktur; amaç yöntemleri adil ve ölçülebilir biçimde kıyaslamaktır.

## Hızlı başlangıç (Windows)

1. Python 3.11 veya 3.12 kurun (kurulumda **Add Python to PATH**).
2. `setup.bat` (sanal ortam, paketler, testler).
3. Veriyi indirin: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption → **Download**. İnen **zip dosyasını açmadan** `data\uci\` klasörüne koyun.
4. `build_db.bat` (ham veriyi SQLite'a yükler, yaklaşık 20-60 sn).
5. `run_experiments.bat` (tüm deneyler; `--quick` ile hızlı deneme). **Okumak için `results\EnerjiRadar_Sonuclar.xlsx` dosyasını açın** (ilk sayfa sütun açıklamaları).
6. `run_app.bat` (web arayüzü).

## Veri

UCI *Individual household electric power consumption*: Fransa'da (Sceaux) tek bir hanenin dakikalık ölçümleri, Aralık 2006 - Kasım 2010 (2.07 milyon satır, %1.25'i eksik). Hebrail & Berard (2012), UCI Machine Learning Repository. Lisans: CC BY 4.0 (UCI sayfasından doğrulayın). Veri repoya konmaz.

## Mimari

```
ham txt/zip -> SQLite (raw_minute) -> SQL view'ları (saatlik / günlük / haftalık)
            -> özellikler (takvim + gecikme, sızıntısız) -> modeller -> kalibrasyon -> metrikler
            -> excess tablosu (SQL) -> Streamlit + Excel
```

```
energyradar/
├── src/energyradar/  data.py (SQL) features.py models.py evaluation.py queries.py pipeline.py report.py config.py
├── app/app.py        Streamlit (6 sayfa)
├── scripts/          build_db.py run_experiments.py
└── tests/test_core.py
```

## Metodoloji

- **Görev:** gün öncesi tahmin. Gün D'nin 24 saati, D 00:00'da (yalnızca D-1 23:00'a kadar bilinen veriyle) tahmin edilir. Özellikler: takvim (saat, haftanın günü, ay, hafta sonu, Fransa resmi tatili), aynı saatin 24/48/168 saat öncesi değeri, önceki günün ve önceki 7 günün ortalaması. Gelecek sızıntısı testle kontrol edilir (gelecek değerleri bozulunca özellikler değişmemeli).
- **Saatlik enerji:** saat içindeki geçerli dakikaların ortalama kW'ı × 1 saat. Saatte 45'ten az geçerli dakika varsa saat **eksik** sayılır ve doldurulmaz.
- **Yürüyen pencere (walk-forward):** 5 test bloğu (2009 sonbahar, 2009-10 kış, 2010 ilkbahar, yaz, sonbahar). Her blokta model yalnızca bloktan önceki verilerle eğitilir; bloğun hemen öncesindeki 60 gün aralık kalibrasyonu için ayrılır (modelin eğitimine girmez). Test verisine bakılarak hiçbir ayar yapılmaz.
- **Modeller:** (1) *Seasonal Naive*: geçen haftanın aynı saati (eksikse dünün aynı saati). (2) *Gradient Boosting*: 5/50/95. yüzdelik için kuantil regresyon, LightGBM (kurulamazsa scikit-learn HistGradientBoosting; Excel'de hangisinin çalıştığı yazar). (3) *LSTM*: son 168 saat + eksiklik maskesi → ertesi günün 24 saati × 3 kuantil, pinball kaybı.
- **Belirsizlik:** %90 aralık. Ham aralık ve **conformal kalibrasyonlu** aralık (CQR; baseline için p50 ± |hata| yüzdeliği) ayrı ayrı kapsama oranıyla raporlanır. Zaman serisinde gözlemler değiştirilebilir (exchangeable) olmadığı için kalibrasyon garantisi yaklaşıktır; sapma ölçülür.
- **Ölçütler:** MAE (birincil), RMSE, WAPE, MAPE (MAPE küçük tüketimlerde şişer, WAPE ile birlikte okunmalı), kapsama oranı, aralık genişliği.
- **Belirsizliğin raporlanması:** 5 rastgele tohum (tohum standart sapması); MAE için **hafta blokları üzerinden bootstrap** %95 güven aralığı ve model çiftleri için fark aralığı (0'ı içeriyorsa fark ayırt edilemez).
- **Koşul kırılımı:** hafta içi / hafta sonu / resmi tatil, mevsim, günün saati.
- **Beklenenden fazla tüketim:** gerçek tüketimin kalibre aralığın **üst sınırını** aştığı saatler (beklenen aşım oranı yaklaşık %5). SQL tablosuna yazılır; toplam fazla kWh, saate/aya dağılım, ardışık aşım olayları (gaps-and-islands) ve alt sayaç karşılaştırması SQL ile çıkarılır.
- **Enjeksiyon testi:** israf tespitinin gerçek doğruluğunu ölçmek için bilinen büyüklükte yapay fazla tüketim (4 ardışık saate +0.25/0.5/1/2 kWh) eklenir ve yakalama oranı Wilson %95 aralığıyla raporlanır. Taban yanlış alarm: eklenmemiş saatlerin aşım oranı.

## Sonuçlar

Tam UCI verisi (2006-2010), 5 yürüyen test bloğu (toplam 65 hafta), 5 tohum, LightGBM + PyTorch LSTM. Ayrıntı: `results/EnerjiRadar_Sonuclar.xlsx`.

| Model | MAE (kWh) | %95 güven aralığı (hafta bootstrap) | WAPE (%) | Aralık kapsama (ham / kalibre) |
|---|---|---|---|---|
| Seasonal Naive (geçen haftanın aynı saati) | 0.566 | [0.533, 0.598] | 52.7 | yok / 0.885 |
| Gradient Boosting (LightGBM) | 0.422 | [0.397, 0.446] | 39.0 | 0.913 / 0.885 |
| LSTM (PyTorch) | 0.422 | [0.399, 0.444] | 39.3 | 0.916 / 0.879 |
| Ensemble (GB + LSTM ortalaması) | 0.413 | [0.389, 0.435] | 38.3 | 0.926 / 0.896 |

- İki öğrenen model de baseline'ı belirgin yeniyor (MAE farkı ≈ -0.144 kWh, aralık [-0.166, -0.122]; ayırt edilebilir).
- **Gradient Boosting ile LSTM arasındaki fark ayırt edilemiyor** (fark -0.0004, aralık [-0.006, +0.006]). **İkisinin ortalaması (ensemble) ikisinden de anlamlı iyi:** GB'ye göre -0.0092 kWh (aralık [-0.0126, -0.0060]), yaklaşık %2. 11 koşulun 8'inde GB'den anlamlı iyi (yaz, resmi tatil ve öğleden sonra hariç).
- **Koşul bazında GB - LSTM (hafta-bloğu bootstrap, 11 koşul):** LSTM hafta sonunda (+0.018) ve gece saatlerinde (+0.010) anlamlı iyi; GB yazın (-0.020) anlamlı iyi. **Kışta fark ayırt edilemiyor** (+0.0097, aralık [-0.0035, +0.0226]), yani "LSTM kışın daha iyi" iddiası desteklenmedi. 11 koşul sınandığı için bu üç sonuç ipucu olarak okunmalıdır (çoklu karşılaştırma). GB baseline'dan 11 koşulun hepsinde anlamlı iyi.
- Hata hafta sonu ve resmi tatilde hafta içine göre yüksek (GB: 0.517 / 0.530 / 0.383 kWh); günün saatine göre en yüksek akşamdadır (0.584).
- **Statik kalibrasyon bu kurulumda kapsamayı iyileştirmedi:** ham aralık hedefe daha yakın (0.913 / 0.916), statik kalibre biraz altında (0.885 / 0.879). **Kayan kalibrasyon (önceki 60 gün)** daha iyi: 0.893 / 0.894, ensemble 0.907. Kalibrasyon penceresi (önceki 60 gün) bazı bloklarda mevsim olarak uyuşmuyor; blok bazında kapsama 0.84-0.94 arası oynuyor.
- **İsraf (beklenenden fazla tüketim):** 530 saat, toplam 234.6 kWh (aşım oranı %4.9, beklenen ~%5). Aşım saatlerinde mutfak ve çamaşırhane alt sayaçları diğer saatlerin yaklaşık 6-8 katı. Bu bir kanıt değil, bir ipucudur.
- **Enjeksiyon testi** (4 saate eklenen yapay fazla tüketim, Gradient Boosting): +0.25 kWh/saat %30.5, +0.5 %53.0, +1.0 %77.5, +2.0 %99.0 yakalandı; yanlış alarm tabanı %5.

## Ek analizler (v2)

Hepsi aynı veriyle ve aynı protokolle çalışır; çalıştırma: `run_experiments.bat`, `run_ablation.bat`, `run_experiments.bat --folds monthly`.

- **Kullanım ritmi özellikleri:** mutfak ve çamaşırhane alt sayaçları için "son kullanımdan beri kaç gün", "son 28 günde kaç gün kullanıldı", "kaç günde bir kullanılıyor (son 8 kullanımın medyan aralığı)" ve "gecikme oranı". Kullanım günü: o gün en az bir saatte alt sayaç ≥ 0.3 kWh (eşik kullanım sıklığına bakılarak seçildi, doğruluğa bakılarak değil). **Not:** veri setinde cihaz bazlı ölçüm yoktur; ütü veya bulaşık makinesi ayrı ölçülmez. Mutfak sayacı (fırın, mikrodalga, bulaşık makinesi), çamaşırhane sayacı (çamaşır/kurutma makinesi, buzdolabı, aydınlatma) cihazların toplamıdır. Su ısıtıcı+klima sayacı günlerin %97'sinde aktif olduğu için ritim taşımaz.
- **Özellik grubu ablasyonu** (`run_ablation.bat`): her grubu tek tek ekler/çıkarır. Kararlar yalnızca ilk 3 blokta (F1-F3) verilir, F4-F5 bağımsız doğrulamadır. Bir ek grup ancak MAE farkının hafta-bloğu bootstrap aralığı tamamen negatifse benimsenir.
- **Ensemble:** Gradient Boosting + LSTM tahminlerinin ortalaması (LSTM çalıştıysa otomatik eklenir).
- **Kayan conformal kalibrasyon:** her test gününün aralığı yalnızca o günden önceki son 60 günün hatalarından hesaplanır; statik kalibrasyonla (üç kapsama sütunu: ham / statik / kayan) birlikte raporlanır.
- **Koşul bazlı anlamlılık:** "LSTM kışın daha iyi mi?" gibi soruları koşul bazında bootstrap ile cevaplar (Excel: "Koşul anlamlılık").
- **Aylık bloklar** (`--folds monthly`): ~34 aylık yürüyen blok, yalnızca baseline + Gradient Boosting; güven aralıklarını daraltır. Sonuçlar `results/monthly/` altında.
- **İsraf alarm kuralları:** "herhangi 1 saat aşım" ve "en az 2 ardışık saat aşım" kuralları yakalama ve yanlış alarm tabanıyla karşılaştırılır; **kalıcı artış testi** (7 gün her akşam +X kWh, tahminler yeniden hesaplanır) ilk alarma kadar geçen günü ölçer.

## Gerçek koşu: aylık bloklar ve ablasyon (kullanıcının bilgisayarı, LightGBM, 3-5 tohum)

- **Aylık bloklar** (34 blok, 148 hafta, baseline + LightGBM): MAE Seasonal Naive 0.589 [0.564, 0.615], Gradient Boosting 0.442 [0.425, 0.460]; fark -0.147 [-0.162, -0.132]. GB, 34 bloğun 33'ünde baseline'dan iyi. 11 koşulun (gün tipi, mevsim, saat dilimi) hepsinde fark ayırt edilebilir. Baseline'a göre iyileşme yıllara göre %21 (2008), %24 (2009), %26 (2010): eğitim verisi arttıkça büyüyor (kısmen az veriyle eğitilen ilk bloklarla açıklanır). Tek kötü blok 2008-08 (GB 0.296, baseline 0.220; kapsama 0.61): alışılmış düzenden sapma, nedeni doğrulanmadı.
- **Ablasyon** (LightGBM, 3 tohum, F1-F3 geliştirme, F4-F5 bağımsız): **kullanım ritmi özellikleri doğruluğu artırmadı** (MAE farkı +0.0006, aralık [-0.0020, +0.0032]); alt sayaç gecikmeleri de katkı vermedi, alt sayaç günlük ortalamaları ve "hepsi" hafifçe kötüleştirdi. Takvim özelliklerini çıkarmak belirgin kötüleştirir (+0.054). Otomatik karar kuralı, gecikmeleri ve günlük ortalamaları birlikte çıkarmaya izin verdi (geliştirme aralığı [-0.003, +0.027], sıfırı içeriyor); ancak F4-F5'te bu seçim temelden belirgin kötü çıktı (0.4125 / 0.3898, fark +0.023, ayırt edilebilir). Yani kural yanlış karar verdi: geniş ve belirsiz bir aralığın sıfırı içermesi "fark yok" kanıtı değildir. Model temel yapılandırmada (takvim + gecikmeler + günlük ortalamalar) kalıyor.

## Dış doğrulama (Portekiz ve Türkiye)

Ana proje tek bir Fransız haneyle kuruldu; aynı yöntem başka verilerde ayrıca sınandı (ana sonuçlarla karıştırılmaz). Çalıştırma: `run_portekiz.bat`, `run_epias.bat` (dosyalar `data\external\` içine konur). Tatil takvimi `holidays` paketinden, enjeksiyon büyüklükleri serinin ortalamasına göre ölçeklenir.

**Portekiz, UCI ElectricityLoadDiagrams20112014 (CC BY 4.0), 5 rastgele müşteri** (tohumlu seçim, 154 uygun müşteriden; 2013-2014, 8 çeyreklik blok, 3 tohum, LightGBM + PyTorch LSTM). Müşteriler büyük ölçekli (ortalama 127-2200 kWh/saat): hane değil, muhtemelen işyeri/sanayi; veri setinde ayrım yoktur.

| Müşteri | Seasonal Naive MAE | LightGBM | LSTM | LightGBM'in iyileşmesi |
|---|---|---|---|---|
| MT_210 | 41.8 | 30.2 | 38.8 | %28 |
| MT_216 | 114.8 | 77.6 | 95.3 | %32 |
| MT_248 | 10.4 | 7.4 | 8.8 | %29 |
| MT_268 | 19.5 | 14.1 | 17.5 | %28 |
| MT_302 | 7.9 | 6.6 | 7.8 | %17 |

- LightGBM baseline'ı 5/5 müşteride anlamlı olarak yendi (hafta-bloğu bootstrap aralığı sıfırı içermiyor).
- LSTM, LightGBM'den 5/5 müşteride anlamlı olarak kötü (~%18-29 daha yüksek hata); baseline'a karşı yalnızca 2/5'te anlamlı iyi. Ensemble (GB+LSTM) LightGBM'i geçmedi (4/5 anlamlı kötü, 1/5 fark yok). LSTM'in geride kalma nedeni sınanmadı (kısıtlı eğitim verisi olası bir açıklama).
- Kalibre aralık kapsaması çoğunlukla 0.87-0.90 (LSTM'de 0.82-0.88), hedef 0.90'ın biraz altında.

**EPİAŞ, Türkiye saatlik tüketim, yalnızca 1 yıl** (2024-11-03 → 2025-11-03; platform daha uzun aralığa izin vermedi): en az 150 gün eğitim, test blokları Mayıs-Ekim 2025 (6 aylık blok). MAE (MWh): Seasonal Naive 2127, LightGBM 1964, LSTM 2668. **LightGBM ile baseline arasındaki fark ayırt edilemiyor** (fark -161, aralık [-847, +444]); LSTM LightGBM'den anlamlı kötü. Temmuz-Ağustos'ta baseline LightGBM'den iyi; eğitimde yaz olmadığı için model görmediği yük aralığına çıkamamış olabilir (sınanmadı). Resmi tatillerde (bayram dahil) LightGBM 2175, baseline 5659 MWh (betimsel; anlamlılık testi için yeterli hafta yok). Aşım oranı %14 (beklenen %5): model gereğinden emin. Bu veride enjeksiyon testi anlamsızdır (büyüklükler ulusal yük için çok büyük, her şey yakalandı); sonuçlara dahil edilmemelidir. Sonuç: 1 yıllık veriyle çıkarım yapılamıyor, bu bir kısıt olarak raporlanır.

## Sınırlılıklar

- **Tek hane, tek ülke.** Sonuçlar bu hane için geçerlidir; sanayiye, işyerine veya Türkiye'ye genellenemez.
- **"Fazla tüketim" israf demek değildir.** Model bilmediği bir davranışı (misafir, ısıtıcı) fazla tüketim olarak işaretler. Sistem yalnızca "tahminin üstünde" der.
- Enjeksiyon testi tek seferlik aşımı ölçer, tahminler temiz veriyle yapılır. Kalıcı bir artış gerçek hayatta gecikmeli özelliklere de girer.
- Kalibrasyon penceresi test bloğundan önceki 60 gündür; mevsim farkı (ör. yaz tatili sonrası) aralığı bozabilir. Sapma kapsama oranında görünür.
- Fransız tatil takvimi yalnızca resmi tatilleri içerir, okul tatillerini içermez.
- Eksik saatler doldurulmaz; bu saatler değerlendirmeye girmez.
- Dış doğrulama: Portekiz müşterileri hane değil; EPİAŞ verisi yalnızca 1 yıl (kısıt yukarıda).
- Birleşik deney tek hane verisi üzerinde yalnızca 5 test bloğu içerir; küçük farklar yorumlanmamalıdır.
