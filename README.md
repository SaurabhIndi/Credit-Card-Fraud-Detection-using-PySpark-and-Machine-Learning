# Credit Card Fraud Detection using PySpark and Machine Learning

## Overview
This project leverages PySpark and machine learning techniques to detect fraudulent credit card transactions in a large-scale dataset. Built with scalability in mind, it uses a Random Forest Classifier to handle imbalanced data and provides performance metrics like AUC, precision, recall, and F1-score. The project is currently under development, with ongoing efforts to refine the model and CI pipeline.

## Features
- Processes large datasets using PySpark for distributed computing.
- Implements SMOTE to address class imbalance in fraud detection.
- Evaluates model performance with cross-validation and various metrics.
- Includes visualizations for confusion matrices and feature importance.

## Technologies
- **PySpark**: For large-scale data processing and machine learning.
- **Python**: Core programming language.
- **scikit-learn**: For additional machine learning utilities.
- **Pandas, Matplotlib, Seaborn**: For data manipulation and visualization.
- **flake8**: For code linting.
- **pytest**: For unit testing.

## Prerequisites
- **Python 3.8 or higher**.
- **Java 11**: Required for PySpark (install via `sudo apt-get install openjdk-11-jdk` on Ubuntu or equivalent).
- **Git LFS**: Optional, for handling the large dataset file (install via [https://git-lfs.github.com/](https://git-lfs.github.com/)).
- **Dependencies**: Install via `pip install -r requirements.txt`.

## Installation

### Clone the Repository
```bash
git clone https://github.com/SaurabhIndi/Credit-Card-Fraud-Detection-using-PySpark-and-Machine-Learning.git
cd Credit-Card-Fraud-Detection-using-PySpark-and-Machine-Learning
```

### Download the Dataset
The project uses the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (143.84 MB). Since it exceeds GitHub's 100 MB limit, it is not included in the repository. Follow these steps:
1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Download the `creditcard.csv` file.
3. Place it in the `data/` directory:
   ```
   data/creditcard.csv
   ```

### Set Up Environment
1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
1. Ensure the `creditcard.csv` file is in the `data/` directory.
2. Run the main script:
   ```bash
   python main.py
   ```
   - **Note**: On Windows, you may see `winutils.exe` warnings. These can be ignored for local runs but require setting `HADOOP_HOME` for production (see [WindowsProblems](https://wiki.apache.org/hadoop/WindowsProblems)).

## Project Structure
- `data/`: Directory for the `creditcard.csv` dataset (user-provided).
- `src/`: Contains Python modules:
  - `preprocessing.py`: Data loading and preprocessing.
  - `model_training.py`: Model building with SMOTE and cross-validation.
  - `evaluation.py`: Model evaluation metrics.
- `tests/`: Directory for unit tests (currently minimal).
- `notebooks/`: Exploratory Data Analysis (EDA) notebook (if included).
- `.github/workflows/`: GitHub Actions CI configuration (currently non-functional).

## Results
- **Test AUC**: ~0.9286 (varies by run).
- **Precision, Recall, F1-Score**: ~0.9993 (under investigation for potential overfitting).
- **Cross-Validation AUC**: 1.0000 (likely indicating an issue with the current setup; see Known Issues).

## Known Issues
- **Perfect Cross-Validation AUC (1.0000)**: The current cross-validation setup may not be splitting data correctly, potentially due to SMOTE application or evaluator misconfiguration. This is under investigation and may require code adjustments.
- **GitHub Actions CI Pipeline**: The `CI Pipeline` workflow is failing (e.g., due to missing tests or linting errors). A basic test file and updated workflow are provided, but further testing and debugging are needed.
- **Memory Warnings**: Running on a local machine may trigger memory-related warnings (e.g., "Not enough space to cache"). Increasing Spark memory settings or sampling the dataset can help.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.
- Please address the known issues or add tests to improve the CI pipeline.

## Future Improvements
- Fix cross-validation to provide realistic AUC scores.
- Optimize memory usage and runtime (e.g., sample dataset for local testing).
- Expand unit tests in the `tests/` directory.
- Resolve GitHub Actions failures and enable code coverage reporting.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- Inspiration and guidance from the xAI community.