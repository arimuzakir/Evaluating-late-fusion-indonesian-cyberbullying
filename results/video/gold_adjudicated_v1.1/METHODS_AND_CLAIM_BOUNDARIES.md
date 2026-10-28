# Batas interpretasi hasil v1.1

Empat label window telah diperbarui berdasarkan keputusan adjudikasi yang dikonfirmasi peneliti. Penilaian awal dipertahankan. Metadata konteks 94 video tidak digunakan sebagai target segmen atau sinyal training.
Encoder teks dan visual dilatih ulang dari pretrained generik secara terpisah untuk setiap seed/fold. Audio waveform16 dan fitur SigLIP2 pretrained generik dipakai kembali karena tidak bergantung pada label penelitian. Korpus komentar tidak dilatih ulang.
Role tetap: fit, early_stop, meta_fit, calibration, test. Pemilihan checkpoint encoder memakai macro F1 .5 di early_stop, earliest tie. Head neural memakai 30 epoch tetap di meta_fit; threshold hanya dipilih di calibration.
Seed 42 untuk inferensi; ketiga seed tetap dilaporkan. Tidak memilih seed atau model menurut skor test tertinggi. Statistik merupakan eksplorasi korpus internal yang sudah diteliti, bukan validasi eksternal baru. Holm terpisah untuk keluarga tujuh kontras utama yang dihitung ulang dan enam kontras eksploratif. Perbandingan representasi lama–baru dari versi label lama tidak digabungkan ke sini.
LOPO hanya probe fitur pretrained dengan penugasan role lama yang dikunci. Ini bukan LOPO end-to-end encoder supervised.
Subset OK-only dan audio-present mempertahankan model/threshold; populasi berbeda, sehingga perbedaan skor subset bukan efek kausal kualitas audio. Audio memiliki 15 deskriptor waveform dan satu indikator ketersediaan, bukan fitur F0.
Komputasi selesai tidak membuktikan identitas/kualifikasi penilai, catatan konsensus historis, atau definisi cyberbullying. Batas tersebut mengikuti deklarasi label v1.1. data_final tetap false sampai audit akhir dan dokumentasi selesai.
