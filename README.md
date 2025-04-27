Credit Card Fraud Detection using PySpark and Machine Learning
Overview
This project detects fraudulent credit card transactions using large-scale data processing with PySpark and machine learning classification.
Technologies

PySpark
Python
scikit-learn
Pandas, Matplotlib, Seaborn

Prerequisites

Install Git LFS to handle large files: https://git-lfs.github.com/
After installing, run:git lfs install





How to Run

Clone the repository:git clone https://github.com/SaurabhIndi/Credit-Card-Fraud-Detection-using-PySpark-and-Machine-Learning.git


Pull the large dataset file using Git LFS:git lfs pull


Set up a virtual environment and install dependencies:pip install -r requirements.txt


Run the project:python main.py



Project Structure

data/: Dataset CSV (tracked with Git LFS).
src/: Code modules (preprocessing, model training, evaluation).
notebooks/: EDA notebook.

Results
Achieved high AUC scores for fraud detection on imbalanced data.
License
This project is licensed under the MIT License.
