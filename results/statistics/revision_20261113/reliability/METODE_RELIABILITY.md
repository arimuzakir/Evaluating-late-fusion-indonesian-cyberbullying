# Analisis tambahan reliability — 12 November 2026

Analisis ini menjawab bagian uncertainty reliability pada R2.6 menggunakan prediksi yang telah tersimpan. Tidak ada training, pemilihan model berdasarkan hasil baru, perubahan label, atau perubahan Word. Seluruh hasil conditional pada model dan Gold v1.1 tersebut.

## Input dan estimand

Input utama adalah CALIBRATION_CONTROL_TEST_PREDICTIONS.csv: 3.951 window, 214 kelompok sumber, seed training 42, lima model (teks, Equal VAT, VAT-LR, sigmoid-calibrated VAT, convex probability pool). BASE_REFERENCE_SEED42.csv merupakan proyeksi tujuh kolom dari OOF historis untuk pemeriksaan silang; hubungan ID kelompok antarfile terbukti bijektif dan tiga kolom probabilitas yang sama cocok hingga toleransi 1e−12. Input beku dan checksum disertakan.

Estimand utama: window eligible dengan bobot sama. Sensitivitas source-weighted memberi setiap window bobot 1/n_g, dengan n_g jumlah seluruh window eligible kelompok g dalam kohort. Karena itu bobot total setiap sumber sama. Bobot tidak dihitung ulang sebagai 1/jumlah anggota sumber di dalam bin. Penetapan hierarki estimand dilaporkan sebagai keputusan revisi, bukan preregistrasi.

## Prosedur

1. Urutkan master_id. Urutan kelompok mengikuti kemunculan pertama setelah pengurutan tersebut.
2. Gunakan bin tetap [0;0,1), [0,1;0,2), …, [0,9;1]. Probabilitas 1 termasuk bin terakhir. Bin tidak disetel untuk mempercantik pola.
3. Hitung mean predicted probability dan observed positive fraction pada setiap model×bobot×bin.
4. Ambil ulang 214 kelompok sumber lengkap dengan pengembalian sebanyak 10.000 replikasi; seed bootstrap 20261112. Implementasi memakai multiplicity multinomial yang ekuivalen dengan pengambilan ulang kelompok. Vektor multiplicity sama digunakan bagi semua model dan kedua pembobotan. Window dalam kelompok selalu ikut bersama.
5. Hitung interval percentile 2,5–97,5% untuk kedua koordinat per bin. Ini interval pointwise, bukan pita simultan atau uji bahwa seluruh kurva berbeda dari diagonal. Grafik menampilkan interval vertikal untuk observed positive fraction; interval mean probability tersedia dalam CSV.
6. Bin dengan n<20 atau kurang dari lima kelompok sumber hanya menampilkan titik terbuka; interval tidak dilaporkan. Bin kosong tidak diplot. Hitung replikasi dengan denominator bin nol; jika kurang dari 95% replikasi mempunyai bin nonkosong, interval ditekan. Bila memenuhi aturan, kuantil hanya memakai replikasi nonkosong.

Dalam hasil ini ada 100 kombinasi: 68 interval, 12 bin jarang tanpa interval, dan 20 bin kosong. Jumlah ini menghitung kedua pembobotan; bukan 100 bin independen. n dan jumlah kelompok asli, jumlah replikasi valid/kosong, batas bin, point estimate dan status interval tersedia di RELIABILITY_POINTWISE_CI.csv.

## Pemeriksaan

Nilai titik dibandingkan dengan RELIABILITY_BINS.csv terdahulu untuk semua lima model pada seed 42. Semua cocok hingga toleransi 1e−12; selisih maksimum numerik kurang dari 4e−16 pada run awal. Tiap model×bobot menjumlah ke 3.951 window. Enam pemeriksaan resampling langsung melalui ekspansi indeks kelompok cocok dengan perhitungan agregat. Pemeriksaan input dan hasil dicatat dalam VALIDATION.json. Script portabel juga diuji ulang; perbandingannya disimpan di verifikasi/PORTABLE_RELIABILITY_CHECK.json.

## Reproduksi

Dari akar folder Bahan_Revisi_Reviewer:

```bash
python analisis_reliability/reproduce_reliability.py --output hasil_reliability_ulang
```

Dependensi: Python, NumPy, pandas dan Matplotlib. Versi yang dipakai terdapat dalam ENVIRONMENT.json. Folder tujuan baru akan dibuat. Opsi --predictions, --reference, dan --existing-bins dapat digunakan untuk jalur lain, tetapi hasil yang berbeda versinya bukan lagi reproduksi input ini. Script menyimpan protokol sebelum sampling dan memeriksa konsistensi input.

Grafik Reliability_Text_EqualVAT berupa dua panel vertikal untuk perbandingan utama. Reliability_All_Five_Models adalah lembar pemeriksaan keseluruhan; untuk publikasi gunakan panel individual atau lampiran dengan ukuran huruf yang memenuhi format jurnal, bukan mengecilkan seluruh lembar ke satu kolom. Ukuran akhir dalam Word belum ditetapkan.

## Teks metode calon integrasi, Bahasa Indonesia

“Reliability dinilai dengan 10 bin probabilitas tetap selebar 0,1, dengan batas bawah inklusif dan batas atas eksklusif kecuali bin terakhir. Interval pointwise 95% dihitung dari 10.000 bootstrap atas kelompok sumber lengkap dengan pengembalian. Multiplicity kelompok yang sama digunakan untuk seluruh model. Analisis utama memberi bobot sama pada window; sensitivitas memberi bobot total yang sama pada setiap kelompok sumber. Bin dengan kurang dari 20 window atau lima kelompok sumber ditampilkan sebagai titik tanpa interval. Ketidakpastian ini conditional pada model terpasang dan tidak memasukkan variasi training maupun perubahan label.”

## Caption calon integrasi, Bahasa Indonesia

“Reliability prediksi teks dan Equal VAT pada window held-out gabungan, seed 42. Panel atas menggunakan bobot window yang sama; panel bawah memberi bobot total yang sama per kelompok sumber. Garis diagonal menunjukkan reliability ideal. Garis vertikal merupakan interval percentile pointwise 95% dari 10.000 bootstrap kelompok sumber. Titik terbuka menunjukkan bin jarang yang tidak diberi interval. Batas bin tetap 0,1; bin kosong tidak ditampilkan. Hasil conditional pada prediksi tersimpan dan bukan ukuran langsung utilitas moderasi.”

Brier paired intervals yang sudah tersedia digunakan bersama grafik ini, bukan dihitung ulang memakai bin. Perubahan peringkat karena pembobotan menggambarkan perbedaan estimand. Tidak ada kesimpulan bahwa ECE/Brier lebih rendah menjamin manfaat aplikasi. Bila label berubah setelah adjudikasi, tentukan dampak training terlebih dahulu lalu regenerasi hasil yang relevan.
