# Breast Cancer Classification using K-Nearest Neighbors (KNN)

This project applies the K-Nearest Neighbors (KNN) machine learning algorithm to classify breast cancer tumors as malignant or benign based on features from the Breast Cancer Wisconsin (Diagnostic) dataset. It demonstrates a simple yet effective supervised learning pipeline for medical diagnosis classification tasks.

## Dataset
The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset, which contains various features computed from digitized images of fine needle aspirate (FNA) of breast masses. Each record is labeled as malignant (M) or benign (B).

## Project Structure and Workflow
- Import essential libraries: numpy, pandas, matplotlib, seaborn, scikit-learn.
- Load and explore the dataset from a CSV file, inspecting structure and summary.
- Preprocess data:
  - Handle missing values (if any).
  - Encode the diagnosis labels (M/B) into numeric format.
  - Normalize feature columns.
- Split the dataset into training and testing sets.
- Train the KNN classifier, tuning the number of neighbors (`n_neighbors`).
- Evaluate model performance using metrics like accuracy, confusion matrix, precision, recall.
- Visualize results using plots such as confusion matrix and accuracy scores.

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/breastCancerDiagnosisUsingKNN.git
```

2. Navigate to the project folder:
```bash
cd breastCancerDiagnosisUsingKNN
```

3. Install required dependencies (preferably in a virtual environment):
```bash
pip install -r requirements.txt
```


4. Run the Jupyter notebook to reproduce the analysis:
```bash
jupyter notebook main.ipynb
```
Or 
convert the notebook to a Python script and run:
```bash
python main.py
```


## Key Libraries and Tools
- Python 3.x  
- NumPy and Pandas for data manipulation  
- Matplotlib and Seaborn for visualization  
- Scikit-learn for machine learning and model evaluation  

## Results
The tuned KNN model achieved high classification performance on the test data with:
- Number of neighbors: 13 (tuned)
- Accuracy: Approximately 97%
- Additional evaluation metrics and confusion matrix plots confirm effective tumor classification.

