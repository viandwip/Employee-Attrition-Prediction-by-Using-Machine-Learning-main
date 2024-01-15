# Employee Attrition Prediction by Using Machine Learning

## 1. Goals & Objectives
**Goals:**
1. Mengurangi tingkat employee attrition (resign) hingga di bawah 10%.
2. Mengetahui faktor-faktor yang dapat menyebabkan seorang karyawan resign.

**Objectives:**
1. Menganalisa faktor-faktor dari data yang dimiliki terhadap potensi employee attrition (resign).
2. Membuat model yang dapat digunakan untuk memprediksi karyawan-karyawan yang berpotensi untuk meninggalkan perusahaan.

## 2. Exploratory Data Analysis
<p align="center"><img src="images/Employee Attrition Ratio.png" alt="Employee Attrition Ratio" width=40%></p>

Berdasarkan ilustrasi di atas, dapat disimpulkan bahwa **16.1%** dari total karyawan memilih untuk **keluar** dari perusahaan sedangkan **83.9%** dari total karyawan memilih untuk **bertahan**.

### 2.1. Univariate Analysis
<p align="center"><img src="images/Univariate Analysis for Numerical Features.png" alt="Univariate Analysis for Numerical Features"></p>

#### Observation:
- Karyawan dengan `Monthly Income` sekitar 1800 - 3200 merupakan tipe karyawan yang memiliki *Attrition Rate* yang paling tinggi. Kemudian karyawan dengan `Monthly Income` sekitar 13000 - 19000 merupakan tipe karyawan yang memiliki *Attrition Rate* yang paling rendah.
- Karyawan dengan rentang umur di bawah 40 tahun cenderung memiliki *Attrition Rate* yang lebih tinggi. Kemudian, karyawan dengan rentang umur sekitar 25 - 35 tahun memiliki *Attrition rate* yang paling tinggi.
- Karyawan dengan nilai `Distance From Home` lebih dari 10 km cenderung memiliki *Attrition Rate* yang lebih tinggi.
- Karyawan dengan nilai `Daily Rate` kurang dari 800 cenderung memiliki *Attrition Rate* yang lebih tinggi.
- Karyawan dengan nilai `Total Working Years` sekitar 0 - 2 tahun serta 4,5 - 6 tahun merupakan dua tipe karyawan yang memiliki *Attrition Rate* yang paling tinggi.
- Karyawan dengan nilai `Years at Company`, `Years in Current`, and `Years with Current Manager` sekitar 0 - 1 tahun tipe karyawan yang memiliki *Attrition Rate* yang paling tinggi.

<p align="center"><img src="images/Univariate Analysis for Categorical Features.png" alt="Univariate Analysis for Categorical Features"></p>

#### Observation:
- Karyawan yang bekerja melebihi jam kerja reguler (`over time`) memiliki *Attrition Rate* yang lebih tinggi dibandingkan dengan karyawan yang tidak bekerja lembur.
- Karyawan yang memiliki tingkat `Job Satisfaction`, `Environment Satisfaction`, `Relationship Satisfaction`, `JobLevel`, `Work Life Balance`, dan `JobInvolvement` yang lebih rendah cenderung memiliki *Attrition Rate* yang lebih tinggi dan yang bernilai 1 merupakan tipe karyawan yang memiliki *Attrition Rate* yang paling tinggi.
- Karyawan yang bekerja pada Departemen `Sales` memiliki *Attrition Rate* yang paling tinggi dibandingkan dengan departemen lainnya.
- Karyawan yang menjabat sebagai `Sales Representative`, `Laboratory Technician`, dan `Human Resources` merupakan 3 tipe karyawan yang memiliki *Attrition rate* paling tinggi.

### 2.2. Bivariate Analysis
<p align="center"><img src="images/Bivariate Analysis.png" alt="Bivariate Analysis"></p>

#### Observation:
Dari grafik di atas yang menunjukkan hubungan `MonthlyIncome` dengan fitur lain (`OverTime`, `JobSatisfaction`,`EnvironmentSatisfaction`, `RelationshipSatisfaction`, `JobLevel`, `WorkLifeBalance`, `JobInvolvement`, `Department`, `JobRole`) dapat disimpulkan bahwa karyawan dengan pendapatan bulanan (`Monthly Income`) yang **lebih rendah** cenderung memiliki *Attrition Rate* yang lebih tinggi atau kecenderungan untuk meninggalkan perusahaan lebih besar.

## 3. Data Preprocessing
1. Tidak terdapat **missing values** atau **data duplikat** pada dataset.
2. Dilakukan penghapusan **outliers** sehingga hanya tersisa **1387** baris data.
3. Merubah fitur kategorik dengan **Label Encoding** jika memiliki 2 *unique values* atau data ordinal dan menggunakan **OHE** jika data nominal.
4. Melakukan **feature engineering** untuk membuat 2 fitur baru, yaitu **EmployeeSatisfaction** dan **JobLevelSatisfaction**.
5. Melakukan **feature selection** menggunakan **Pearson Correlation**.
6. **Standarisasi** untuk data training dan test.
7. Menangani **class imbalance** dengan menggunakan teknik **SMOTE**.
8. Membagi data dengan proporsi 70:30, **70% untuk training** dan **30% untuk testing**.

## 4. Modeling
### 4.1. Model Training & Validation
Metrics yang akan digunakan untuk mengukur tingkat keberhasilan model *machine learning* adalah *metric* **recall**. Hal ini karena **false negative** memiliki dampak negatif yang lebih besar kepada perusahaan seperti terjadinya kekosongan jabatan sehingga dapat berpotensi menurunkan performa perusahaan ketimbang **false positive**.

| No | Model | Recall (Train) | Recall (Test) | Cross Validation (Train) | Cross Validation (Test) | Time Elapsed
| :- | :- | :- | :- | :- | :- |:- |
| 1 | KNNeighbors | 0.992565 | 0.971510 | 0.997217 | 0.989633	| 1.431684
| 2 | Extra Trees |	1.000000 | 0.914530 | 1.000000 | 0.922391	| 2.786827
| 3 | Random Forest | 1.000000 | 0.871795 | 1.000000 | 0.908598 | 4.175266
| 4 | Ada Boost | 0.912020 | 0.874644 | 0.902707	| 0.868028 | 2.066318
| 5 | Gradient Boosting | 0.950434 | 0.871795 | 0.931108	| 0.867226 | 6.598724
| 6 | Decision Tree	| 1.000000 | 0.840456 | 1.000000 | 0.835210	| 0.293376
| 7 | Logistic Regression	| 0.843866 | 0.846154 | 0.848014 | 0.832534 | 0.216894 

Berdasarkan hasil **Cross Validation (Test)** di atas, maka tiga model terbaik yang dipilih untuk *hyperparameter tuning* adalah model **KNNeigbors**, **Extra Trees** dan **Random Forest**.

### 4.2. Hyperparameter Tuning

| No | Model | Recall (Train) | Recall (Test) |
| :- | :- | :- | :- |
| 0	| Random Forest |	0.997522 | 0.840456
| 1	| KNNeighbors	| 0.975217	| 0.937322
| 2	| Extra Trees	| 0.996283	| 0.883191

Berdasarkan hasil *hyperparameter tuning* di atas, didapatkan bahwa **model KNNeighbors** adalah model terbaik karena memiliki skor **recall paling tinggi** serta perbedaan skor *recall* pada *data train* dan *data test* yang tidak terlalu jauh, sehingga sudah dapat dikatakan **best-fit** atau tidak terjadi *overfitting*.

### 4.3. Confusion Matrix
<p align="center"><img src="images/Confusion Matrix.png" alt="Confusion Matrix"></p>

Dengan menggunakan hasil *hyperparameter tuning* untuk model KNN, kita melatih lagi model tersebut untuk mendapatkan **confusion matrix** seperti gambar di atas, dengan hasil yaitu:

- **True Positive**: Diprediksi keluar dan ternyata benar sebanyak 329
- **True Negative**: Diprediksi bertahan dan ternyata benar sebanyak 240
- **False Positive**: Diprediksi keluar dan ternyata salah sebanyak 104
- **False Negative**: Diprediksi bertahan dan ternyata  salah sebanyak 22

**Employee attrition rate** setelah menggunakan *machine learning*: (22/695) * 100% = **3.2%**

### 4.4. SHAP Values
<p align="center"><img src="images/SHAP Values.png" alt="SHAP Values" width=70% ></p>

### 4.5. Business Insights
Beberapa insight yang bisa didapatkan berdasarkan grafik **Shap Values** pada slide sebelumnya adalah :
1. Sepuluh fitur yang paling mempengaruhi tingkat attrition karyawan adalah `EmployeeSatisfaction`, `StockOptionLevel`, `DistanceFromHome`, `YearsInCurrentRole`, `WorkLifeBalance`, `JobLevelSatisfaction`, `TotalWorkingYears`, `Age`, `MaritaslStatus_Married`, and `DailyRate`.
2. **Semakin rendah** tingkat kepuasan seorang karyawan terhadap perusahaan semakin tinggi kemungkinan karyawan tersebut untuk keluar dari perusahaan.
3. **Semakin rendah** kompensasi seorang karyawan dalam bentuk ekuitas semakin tinggi kemungkinan karyawan tersebut untuk keluar dari perusahaan.
4. **Semakin jauh** jarak seorang karyawan dengan perusahaan semakin tinggi kemungkinan karyawan tersebut untuk keluar dari perusahaan.
5. **Semakin rendah** jumlah tahun seorang karyawan bekerja pada posisi yang sedang dikerjakan semakin tinggi kemungkinan karyawan tersebut untuk keluar dari perusahaan.
6. **Semakin rendah** tingkat *work life balance* dari seorang karyawan semakin tinggi kemungkinan karyawan tersebut untuk keluar dari perusahaan.
7. **Semakin rendah** tingkat kepuasan dari seorang karyawan terhadap level posisi pekerjaannya semakin tinggi kemungkinan karyawan tersebut untuk keluar dari perusahaan.
8. **Semakin rendah** pengalaman kerja dari seorang karyawan semakin tinggi kemungkinan karyawan tersebut untuk keluar dari perusahaan.
9. **Semakin rendah** muda umur seorang karyawan semakin tinggi kemungkinan karyawan tersebut untuk keluar dari perusahaan.
10. Karyawan dengan *marital status* menikah memiliki *attrition rate* yang **lebih rendah** jika dibandingkan dengan karyawan dengan marital status selain menikah.
11. **Semakin tinggi** *Daily Rate* maka semakin tinggi juga kemungkinan karyawan untuk keluar dari perusahaan.

### 4.6. Business Recommendations
1. Melakukan peninjauan kembali terhadap fasilitas yang sudah ada atau yang akan diberikan kepada masing-masing karyawan seperti **gaji** dan **insentif** untuk meningkatkan *job satisfaction*.
2. Menerapkan **positive culture** sebagai budaya perusahaan serta membuat sebuah kegiatan untuk meningkatkan *engagement* antar karyawan untuk meningkatkan *environment satisfaction* para karyawan.
3. Kami merekomendasikan karyawan diberikan prioritas dan kesempatan untuk terlibat dalam **pembelian saham** perusahaan.
4. Menerapkan **hybrid working** bagi karyawan yang memiliki rumah jauh dari kantor.
5. Kami merekomendasikan untuk memberikan fasilitas bagi karyawan untuk melakukan **self-development** atau **training**, serta pemberian **jenjang karir yang jelas** kepada karyawan muda atau yang baru masuk sehingga bisa meningkatkan *job level* dan lama seorang karyawan bekerja di suatu jabatan tertentu.
6. Melakukan **analisa beban kerja** masing-masing karyawan untuk meminimalisir jumlah karyawan yang bekerja diluar jam kerja reguler (*over time*) dengan memperhitungkan jumlah karyawan yang berada pada masing-masing departemen untuk meningkatkan *work-life balance* karyawan.
