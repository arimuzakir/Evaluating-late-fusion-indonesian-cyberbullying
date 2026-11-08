# Catatan notebook 12 revisi CPU v4

Tidak ada training atau ekstraksi media baru. Prediksi GPU dari run asli dipertahankan dan diperiksa checksum-nya.

| Poin | Tambahan | Batas yang dipertahankan |
|---|---|---|
| R1.4 | Confusion matrix fallback 21 model × 3 seed × pooled/4 fold; keputusan fallback teks dijelaskan | Hasil memakai prediksi dan threshold tersimpan |
| R1.7/R1.8 | MDE memakai SE bootstrap asli 0,0075124837897468; pendekatan normal menghasilkan 0,0210468635366681 | Diagnostik retrospektif kondisional; bukan power prospektif, Holm-adjusted MDE atau ekuivalensi |
| R1.1/R1.3 | Tabel model–role, arsitektur head, alokasi dan keterbatasan runtime | Gold mengatur presisi matriks high; run tambahan tidak merekam nilai efektif. Penyebab selisih skor belum diisolasi |
| R2.5 | Ledger membedakan partisi 2042 dari effective outer seed 2044 | Pemilihan kandidat menggunakan kelas/platform, bukan skor model |
| R1.5 | Replay sampling cocok; batas hipergeometrik berstrata konservatif tersedia, termasuk sel singleton | Target adalah proporsi fidelity dalam korpus tetap, bersyarat pada desain dan rating adjudikasi; bukan generalisasi ke sumber baru |
| R2.3 | Hitungan transisi error dan data numerik berbobot dapat diperiksa | CI transisi subkelompok tetap NA; mekanisme fusion belum terjawab |
| R2.6 kontrol | Dua kontrol seed 42: CI Brier versus text dan Equal VAT, tabel probabilitas Tabel 6, dan reliability berbobot window/sumber | Prediksi arsip dan role asli dipertahankan; interval deskriptif tanpa koreksi multipel dan tanpa refit |
| R2.6 | Window-weighted dipilih eksplisit untuk tugas window 10 detik; source-weighted menjadi sensitivitas | Keputusan pelaporan saat revisi, tidak diklaim preregistered |

Interval fidelity utama memakai inversi dua ekor hipergeometrik per strata dan koreksi Bonferroni untuk seluruh 20 branch-stratum outcomes. Batas jumlah sukses dijumlahkan untuk mendapatkan batas proporsi. Cakupan konservatif setidaknya 95% berlaku atas sampling window tanpa penggantian dari populasi tetap, dengan rating tetap. Pola kesamaan outcome dalam sumber tidak diasumsikan iid. Ini tidak menciptakan interval generalisasi ke video baru atau mengukur kesalahan penilai. Interval model-assisted dengan multiplier sumber tetap dilaporkan terpisah sebagai analisis eksploratif. Batas konservatif dapat lebar; tidak dipilih menurut interval mana yang lebih mendukung kesimpulan.

Faithful OCR tetap 43/52 dan STT 51/64. Penarikan interval transisi error tidak mengubah hitungan atau prediksi. Original Table 3 dan hasil sensitivitas lingkungan baru tetap dipisahkan.

Tambahan kontrol tidak mengubah hasil utama atau hasil tiga seed. Dua kontrol hanya tersedia untuk seed 42. Grafik reliability tetap deskriptif tanpa CI per bin.

GitHub dan naskah belum diedit. Versi ekstraksi historis, penetapan empat kontras akhir, serta penetapan institusi tetap memerlukan bukti yang sesuai.
