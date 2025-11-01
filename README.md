# credit-card-fraud-detection
A machine learning project to detect fraudulent credit card transactions. This repository contains the code for data processing, model training (e.g., Logistic Regression, Random Forest), and evaluation.
# credit-card-fraud-detection
A machine learning project to detect fraudulent credit card transactions. This repository contains the code for data processing, model training (e.g., Logistic Regression, Random Forest), and evaluation.

# 💳 Credit Card Fraud Detection Project

---

## 1. Objective

The primary goal of this project is to develop a high-performance machine learning model capable of accurately identifying fraudulent credit card transactions. The model leverages pattern recognition and anomaly detection techniques on a highly imbalanced dataset to minimize financial losses by effectively flagging suspicious activities.

---

## 2. Tech Stack & Environment

- **Programming Language:** Python
- **Libraries & Frameworks:**
  - **Data Manipulation:** Pandas, NumPy
  - **Visualization:** Matplotlib, Seaborn
  - **Machine Learning:** Scikit-learn (Logistic Regression, Random Forest), XGBoost
  - **Imbalanced Data Handling:** Imbalanced-learn (SMOTE)
- **Environment:** Jupyter Notebook / Google Colab

---

## 3. Project Workflow

The project was structured into a series of logical steps, from data exploration to final model optimization.

### 🔹 Step 1: Exploratory Data Analysis (EDA)
The initial phase involved a deep dive into the Kaggle credit card fraud dataset. Key activities included:
- Analyzing the severe class imbalance between fraudulent (Class 1) and non-fraudulent (Class 0) transactions.
- Visualizing transaction patterns and feature distributions.
- Studying the statistical differences in transaction `Amount` and `Time` between the classes.

### 🔹 Step 2: Data Preprocessing
Before modeling, the data was prepared for machine learning algorithms:
- **Feature Scaling:** The `Time` and `Amount` columns were normalized using `StandardScaler`.
- **Data Splitting:** The dataset was split into training (80%) and testing (20%) sets, using stratification to maintain the class distribution.

### 🔹 Step 3: Handling Class Imbalance
To address the severe imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied *only* to the training data. This creates a balanced dataset for the models to learn from without causing data leakage.

### 🔹 Step 4: Baseline Model Training
Three different classification models were trained on the balanced (resampled) training data to establish a performance baseline:
1.  Logistic Regression
2.  Random Forest Classifier
3.  XGBoost Classifier

### 🔹 Step 5: Hyperparameter Tuning
The best-performing baseline model (XGBoost) was selected for optimization. **`RandomizedSearchCV`** was used to efficiently search for the best hyperparameters, optimizing for the ROC AUC score.

---

## 4. Results & Evaluation

The models were rigorously evaluated on the original, untouched test set. The primary metrics were **Recall** and **Precision** for the minority (fraud) class.

- **Baseline Models:** All models performed reasonably well, with Random Forest and XGBoost showing superior performance in identifying fraudulent transactions.
- **Optimized Model:** After hyperparameter tuning, the final XGBoost model showed significant improvement.

**Final Tuned XGBoost Model Performance:**
*(From notebook `03_Hyperparameter_TTuning.ipynb`)*
- **Precision (Class 1):** `0.77`
- **Recall (Class 1):** `0.88`
- **F1-Score (Class 1):** `0.82`
- **ROC AUC Score:** `0.9811`

The outcome is a robust fraud detection model with high recall, ensuring that a vast majority of fraudulent transactions are caught.

---

## 5. How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nidhirautioit/credit-card-fraud-detection.git](https://github.com/nidhirautioit/credit-card-fraud-detection.git)
    cd credit-card-fraud-detection
    ```

2.  **Download the Dataset:**
    * The dataset (143 MB) is too large for GitHub.
    * Download it from the official Kaggle page:
        **[Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**
    * **Important:** After downloading, place the `creditcard.csv` file inside the `data/` folder in the project.

3.  **Create a virtual environment**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    source venv/bin/activate # On macOS/Linux
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

5.  **Run the notebooks:**
    * Open the `notebooks` directory.
    * Run the notebooks in sequential order:
        1.  `01_EDA_and_Preprocessing.ipynb`
        2.  `02_Model_Training_and_Evaluation.ipynb`
        3.  `03_Hyperparameter_Tuning.ipynb`
