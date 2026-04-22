# Clay Classification using Machine Learning

## Objective
The objective of this project is to classify different clay types based on experimentally measured physicochemical properties such as particle size distribution, chemical composition, and physical characteristics.

---

## Workflow / Pipeline

1. **Data Generation**  
   Experimental dataset collected from different clay samples.

2. **Preprocessing**  
   - Feature scaling (Standardization)  
   - Label encoding for clay categories  

3. **Modeling**  
   - Random Forest  
   - Support Vector Machine (SVM)  

4. **Evaluation**  
   - Accuracy score  
   - Confusion matrix  
   - Classification report  

5. **Visualization**  
   - Confusion matrix heatmap  
   - Feature importance (Random Forest)  

6. **Robustness Check**  
   - Noise perturbation added to input features to simulate measurement uncertainty  

---

## Dataset Description

The dataset consists of experimental measurements of clay samples, including:

- Particle size distribution: d10, d50, d90, span  
- Chemical composition: SiO₂, Al₂O₃, Fe₂O₃, CaO, MgO, LOI  
- Derived features: Al/Si ratio  
- Physical properties: water absorption  

Each sample is labeled with its corresponding clay type.

---

## Repository Structure

```
clay-classification-ml/
│
├── data/
│   └── clay_data.csv
│
├── notebooks/
│   └── clay_classification.ipynb
│
├── README.md
└── requirements.txt
```
## How to Run

1. Clone or download this repository  
2. Install required libraries:
## pip install -r requirements.txt
3. Open the notebook:
## notebooks/clay_classification.ipynb
4. Run all cells sequentially  

---

## Expected Outputs

- Model performance metrics (accuracy, classification report)  
- Confusion matrix visualization  
- Feature importance plot  
- Robustness evaluation results  

---

## Assumptions and Limitations

- Dataset size is limited to available experimental samples  
- Model performance depends on quality and variability of input features  
- Synthetic noise used in robustness check approximates measurement uncertainty  

---

## Summary

This project demonstrates how machine learning techniques can be applied to classify clay materials based on their physicochemical properties, providing a data-driven approach to material characterization.
