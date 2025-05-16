# Workflow-CI for Automated Model Training

Repo ini berisi workflow CI menggunakan MLflow Project untuk melakukan training dan tuning model secara otomatis ketika trigger dijalankan.

Struktur folder:
- MLProject/: berisi script training (modelling.py), environment (conda.yaml), konfigurasi MLflow (MLproject), dataset preprocessing, dan Dockerfile.
- .github/workflows/: berisi workflow GitHub Actions yang menjalankan MLflow Project otomatis.

Cara menggunakan:
- Pastikan file dataset `student-depression-dataset_preprocessing.csv` tersedia di folder MLProject/
- Push ke branch main, workflow akan otomatis menjalankan training model.
