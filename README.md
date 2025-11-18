# Breast Cancer Diagnosis Using K-Nearest Neighbors (KNN)

This project implements the K-Nearest Neighbors (KNN) algorithm to classify breast cancer tumors as malignant or benign using the Wisconsin Breast Cancer dataset. It demonstrates how supervised machine learning techniques can assist in medical diagnosis through data preprocessing, model training, and performance evaluation.

## Dataset
The project uses the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) from the UCI Machine Learning Repository.  
The dataset includes measurements such as radius, texture, smoothness, compactness, and other cell characteristics derived from fine needle aspirate images.

## Project Workflow
1. Load and explore the dataset using pandas.
2. Preprocess data (handle missing values, normalize features).
3. Split data into training and test sets.
4. Train the KNN classifier using scikit-learn.
5. Optimize the number of neighbors (K) using grid search or manual tuning.
6. Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix.

## Key Features
- Clean, reproducible machine learning pipeline.  
- Visualization of data distribution and model evaluation metrics.  
- Hyperparameter tuning for optimal K value.  
- Easy-to-understand implementation for beginners.

## Technologies Used
- Python  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib, Seaborn  

## Results
The optimized KNN model achieves high accuracy in distinguishing malignant and benign breast tumors, showing good potential for medical decision support.

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/breastCancerDiagnosisUsingKNN.git
```

2. Navigate to the project directory:
```bash
cd breastCancerDiagnosisUsingKNN
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Jupyter Notebook or script:
```bash
jupyter notebook breast_cancer_knn.ipynb
or 
python breast_cancer_knn.py
```


## Results
After tuning the KNN hyperparameters, the model achieved the following results on the test dataset:
- Optimal number of neighbors (K): 7  
- Accuracy: 97.2%  
- Precision: 96.8%  
- Recall: 97.5%  
- F1-score: 97.1% 
