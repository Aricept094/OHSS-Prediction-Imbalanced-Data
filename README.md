# Prediction of Complicated Ovarian Hyperstimulation Syndrome (OHSS) using Machine Learning

This repository contains the Python code used for the research paper titled: **"Prediction of Complicated Ovarian Hyperstimulation Syndrome in Assisted Reproductive Treatment Through Artificial Intelligence"** by Arash Ziaee, Hamed Khosravi, Tahereh Sadeghi, Imtiaz Ahmed, and Maliheh Mahmoudinia.

## Abstract (from paper)

**Background:** This study explores the utility of machine learning (ML) models in predicting complicated Ovarian Hyperstimulation Syndrome (OHSS) in patients undergoing infertility treatments, addressing the challenge posed by highly imbalanced datasets.
**Objective:** This research fills the existing void by introducing a detailed structure for crafting diverse machine learning models and enhancing data augmentation methods to predict complicated OHSS effectively. Importantly, the research also concentrates on pinpointing critical elements that affect OHSS.
**Method:** This retrospective study employed a ML framework to predict complicated OHSS in patients undergoing infertility treatment. The dataset included various patient characteristics, treatment details, ovarian response variables, oocyte quality indicators, embryonic development metrics, sperm quality assessments, and treatment specifics. The target variable was OHSS, categorized as painless (no OHSS), mild, moderate, or severe, then binarized into Uncomplicated (0) and Complicated (1). The ML framework incorporated Ray Tune for hyperparameter tuning and SMOTE-variants for addressing data imbalance. Multiple ML models were applied and integrated into a voting classifier. The SHAP package was used for interpretation.
**Results:** The best model utilized IPADE-ID augmentation along with an ensemble of classifiers (Stochastic Gradient Descent, Support Vector, and Ridge Regression), achieving a recall of 0.9 for predicting complicated OHSS occurrence and an accuracy of 0.76. SHAP analysis identified key factors including GnRH antagonist use, longer stimulation, female infertility factors, irregular menses, higher weight, hCG triggers, and higher number of embryos.
**Conclusion:** This study demonstrates ML's potential for predicting complicated OHSS. The optimized model provides insights into contributory factors, challenging certain conventional assumptions and highlighting the importance of patient-specific factors and treatment details.

## Publication

[Link to Publication - To be added upon publication/preprint availability]

## Requirements

This project was developed using Python 3.9. Key dependencies include:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `lightgbm`
*   `xgboost`
*   `ray[tune]`
*   `smote-variants`
*   `scipy`
*   `joblib`
*   `pyarrow`
*   `optuna` (for Ray Tune search algorithm)
*   `wandb` (Optional, for logging - currently commented out in script)
*   `aim` (Optional, for logging)
*   `shap` (Used for interpretation as mentioned in the paper, although not explicitly in the provided optimization script)

It is highly recommended to use a virtual environment (`venv` or `conda`). You can install the required packages using pip:

```bash
pip install -r requirements.txt
