# Prediction of Complicated Ovarian Hyperstimulation Syndrome using Machine Learning

This repository contains the Python code for the study titled: **"Prediction of Complicated Ovarian Hyperstimulation Syndrome in Assisted Reproductive Treatment Through Artificial Intelligence"**.

## Description

Ovarian Hyperstimulation Syndrome (OHSS) is a significant complication in Assisted Reproductive Technology (ART). Predicting *complicated* OHSS (moderate/severe cases requiring intervention) is challenging, particularly due to the highly imbalanced nature of clinical datasets.

This project implements a comprehensive machine learning framework designed to:
1.  Predict the likelihood of complicated OHSS in patients undergoing infertility treatments.
2.  Address severe class imbalance using advanced data augmentation techniques (specifically exploring variants from the `smote-variants` library).
3.  Systematically optimize the entire prediction pipeline (preprocessing, feature selection, model selection, and hyperparameters) using Ray Tune.
4.  Identify key clinical factors contributing to complicated OHSS risk using SHAP (SHapley Additive exPlanations).

The framework explores various ML models (including Logistic Regression, SVM, SGD, Ridge Regression, KNN, Tree-based models) and integrates them into an ensemble Voting Classifier. The optimization process aims to maximize recall for the minority class (complicated OHSS) while maintaining reasonable overall performance.

The best model identified in the associated study utilized **IPADE-ID** for data augmentation combined with an **ensemble of Stochastic Gradient Descent, Support Vector Machine, and Ridge Regression classifiers**, achieving high recall (0.9) for complicated OHSS prediction.

## Related Publication

**Title:** Prediction of Complicated Ovarian Hyperstimulation Syndrome in Assisted Reproductive Treatment Through Artificial Intelligence
**Authors:** Arash Ziaee¹, Hamed Khosravi², Tahereh Sadeghi³, Imtiaz Ahmed⁴, Maliheh Mahmoudinia⁵*
(*Affiliations listed in the paper*)
**Status:** (e.g., Submitted / Under Review / Published in [Journal Name] / Link to Preprint) - *[Please update this when available]*

Please cite the publication if you use this code or research findings.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [URL of your GitHub repository]
    cd [repository-name]
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The required libraries are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include: `pandas`, `numpy`, `scikit-learn`, `ray[tune]`, `smote-variants`, `optuna`, `xgboost`, `lightgbm`, `shap`, `joblib`, `pyarrow`. *(Ensure `requirements.txt` is generated and included in the repo)*.

    **Important Note on `smote-variants`:**

    To ensure reproducibility of the results presented in this work, a specific modification was required for the `smote-variants` library. The standard version (tested with version 1.0.0) contains an issue where the internal `model_selection` routine fails when evaluating parameter sets containing NumPy data types     (e.g., `np.False_`, `np.str_`) due to its use of         `ast.literal_eval`. therefore the requiremts.txt downloads a custom version with fix from the follwoing repo : https://github.com/Aricept094/smote_variants_literal_eval_fix_for_Ray.git@main#egg=smote_variants

5.  **Python Version:** Developed and tested using Python 3.x. *(Specify exact version if known, e.g., 3.9)*

## Data

**IMPORTANT:** Due to patient privacy regulations and the ethics approval (Mashhad University of Medical Sciences ethics committee IR.MUMS.REC.1395.326), the **original patient dataset cannot be shared publicly** in this repository.

The script (`tune_ohss_pipeline.py`) expects input data in the following format:
*   Two separate Excel files: `train_data.xlsx` and `test_data.xlsx`.
*   These files should be placed in a directory specified by the hardcoded paths `train_data_path` and `test_data_path` within the script (currently `/home/[USEER_NAME]/my_data/`). **You will need to modify these paths.**
*   The files should contain the features listed in `feature_list` within the script (corresponding to Table 1 in the paper) and the target variable `OHSS`.
*   **Target Variable (`OHSS`):** Encoded as 0 for Uncomplicated OHSS (Painless/Mild) and 1 for Complicated OHSS (Moderate/Severe).
*   **Data Preprocessing:** The script assumes the input data has already been handled for missing values (e.g., using imputation methods like those described in the paper - Random Forest for continuous, mean for categorical).

**For testing purposes:** You may want to create a small, synthetic, or anonymized dummy dataset following the expected structure and column names to run the script and verify its functionality.

## Usage

1.  **Modify Paths:** Open `tune_ohss_pipeline.py` and update the `train_data_path`, `test_data_path`, and `log_directory_base` variables to point to your data location and desired output directory.
2.  **Run the Script:** Execute the main script from your terminal within the activated virtual environment:
    ```bash
    python tune_ohss_pipeline.py
    ```
3.  **Process:** This script will initiate a Ray Tune hyperparameter optimization process, running a large number of trials (defined by `num_samples=15000`) to find the best combination of preprocessing steps, feature subsets, data augmentation techniques (SMOTE variants), models, and hyperparameters based on the `recall_mean` metric.
4.  **Output:**
    *   The script will create a main log directory (`log_directory_base`) named with the execution timestamp.
    *   Inside, it will create subdirectories for each Ray Tune trial.
    *   Each trial directory will contain logs, saved intermediate dataframes (`.csv`), configuration details, and potentially saved unfitted/fitted model files (`.joblib`).
    *   The console will output progress, and the best configuration found during the run will be printed at the end.

## Reproducing Results

Running `tune_ohss_pipeline.py` executes the hyperparameter search framework described in the paper. The goal is to find high-performing pipeline configurations for predicting complicated OHSS.

*   The script explores the `search_space` defined within it.
*   The best configuration printed at the end represents the optimal pipeline found in that specific run, maximizing the custom `recall_mean` metric.
*   Due to the stochastic nature of the search and model training, the *exact* best configuration might vary slightly between runs.
*   The configuration reported in the paper (IPADE-ID augmentation + SGD/SVC/Ridge ensemble) was identified through this optimization process and represents one such high-performing result achieving Recall=0.9 for Class 1 and Accuracy=0.76.

## Citation

If you use this code or the findings from the associated study in your research, please cite both the original paper and this software repository.

**Paper:**
*   Ziaee, A., Khosravi, H., Sadeghi, T., Ahmed, I., & Mahmoudinia, M. (Year). Prediction of Complicated Ovarian Hyperstimulation Syndrome in Assisted Reproductive Treatment Through Artificial Intelligence. *[Journal Name/Conference Proceedings, Volume, Pages, DOI - Update when published]*

**Software:**
*   Please use the citation information provided in the `CITATION.cff` file or the "Cite this repository" button on GitHub.

## License

This project is licensed under the MIT. See the `LICENSE` file for details. *
## Contact

For questions regarding the research or code, please contact the author:
*   Arash Ziaee : `ziaeia961@mums.ac.ir`

---
