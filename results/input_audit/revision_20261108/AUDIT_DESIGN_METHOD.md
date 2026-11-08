# Metode batas fidelity pada populasi tetap

Pada strata h, N_h dan n_h diketahui, k_h adalah jumlah rating faithful yang diamati, dan K_h adalah jumlah faithful tak diketahui dalam seluruh strata. Sampling yang berhasil direproduksi adalah SRSWOR per availability × platform, sehingga k_h mengikuti Hypergeom(N_h,K_h,n_h) atas pengacakan desain, dengan nilai outcome populasi diperlakukan tetap.

Setiap batas K_h dibangun dengan menginversi kedua ekor hipergeometrik pada alpha/(2M), dengan M=20 untuk seluruh branch-stratum outcomes OCR/STT yang tersedia. Union bound memberi cakupan serentak sekurang-kurangnya 1-alpha tanpa membutuhkan independensi antar-outcome. Penjumlahan batas K_h dan pembagian dengan jumlah N_h pada strata tersedia menghasilkan batas konservatif untuk setiap proporsi branch. Ini konstruksi konservatif yang transparan, bukan implementasi metode optimal untuk nuisance parameters.

Unit yang diacak adalah window, bukan cluster sumber. Karena ini inferensi populasi tetap, pola outcome antar-window dalam sumber boleh sembarang; tidak ada asumsi outcome iid atau superpopulasi sumber. Interval tidak menyatakan ketidakpastian transfer ke sumber baru. Sensitivitas sumber memakai multiplier bersama per sumber tetap bersifat model-assisted dan tidak boleh diberi label design-exact. Rating adjudikasi diperlakukan sebagai pengukuran tetap dan kesalahan pengukurannya tidak dicakup.

Rujukan konstruksi inversi dan komputasi:

- https://ucb-stat-159-s21.github.io/site/Notes/binom.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html

Reproduksi numerik: baca AUDIT_NUMERIC_ROWS.csv dan AUDIT_SAMPLING_ALLOCATION.csv, hitung k_h=(accuracy==2) per strata tersedia, jalankan finite_success_bounds, lalu jumlahkan count endpoints. Semua kolom yang diperlukan tersedia tanpa teks, media atau identitas sumber asli.
