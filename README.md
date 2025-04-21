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

1.	General Physician, Student Research Committee, Faculty of Medicine, Mashhad University of Medical Sciences, Mashhad, Iran
2.	Department of Industrial & Management Systems Engineering, West Virginia University, Morgantown, WV, US
3.	Assistant Professor of Nursing, Department of Pediatrics, School of Nursing and Midwifery, Nursing and Midwifery Care Research Center, Akbar Hospital, Mashhad University of Medical Sciences, Mashhad, Iran
4.	Assistant Professor, Department of Industrial & Management Systems Engineering, West Virginia University, Morgantown, WV, US
5.	Assistant Professor of Obstetrics & Gynecology, Fellowship of Infertility, Supporting the Family and the Youth of Population Research Core, Department of Obstetrics and Gynecology, Faculty of Medicine, Mashhad University of Medical Sciences, Mashhad, Iran - Coresponding Author

**Status:** (Currently Under Review / Link to Preprint) - (https://www.medrxiv.org/content/10.1101/2024.04.17.24305980v1)

Please cite the publication if you use this code or research findings.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Aricept094/OHSS-Prediction-Imbalanced-Data.git
    cd OHSS-Prediction-Imbalanced-Data
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

4.  **Python Version:** Developed and tested using Python 3.10

## Data

**IMPORTANT:** Due to patient privacy regulations and the ethics approval (Mashhad University of Medical Sciences ethics committee IR.MUMS.REC.1395.326), the **original patient dataset cannot be shared publicly** in this repository.

The script (`tune_ohss_pipeline.py`) expects input data in the following format:
*   Two separate Excel files: `train_data.xlsx` and `test_data.xlsx`.
*   These files should be placed in a directory specified by the hardcoded paths `train_data_path` and `test_data_path` within the script (currently `/home/[USEER_NAME]/my_data/`). **You will need to modify these paths.**
*   **Required Features:** The input data files (`train_data.xlsx`, `test_data.xlsx`) **must** contain the target variable `OHSS` and **all** of the following features (column names), exactly as listed. The script **will fail** if any of these columns are missing or named differently. Descriptions are based on the associated study:

    *   `age`: Patient's age at baseline (Years).
    *   `weight`: Patient's weight at baseline (Kilograms).
    *   `Height`: Patient's height at baseline (Centimeters).
    *   `Durationinfertility`: Duration of infertility at baseline (Years or relevant time unit).
    *   `FSH`: Follicle-Stimulating Hormone level at baseline (IU/L).
    *   `LH`: Luteinizing Hormone level at baseline (IU/L).
    *   `numberofcunsumptiondrug`: Total number of stimulation drug doses consumed during the cycle.
    *   `durationofstimulation`: Duration of ovarian stimulation (Days).
    *   `numberRfulicule`: Count of follicles in the right ovary on the day of hCG trigger.
    *   `numberLfulicule`: Count of follicles in the left ovary on the day of hCG trigger.
    *   `numberofoocyte`: Total count of oocytes retrieved on egg retrieval day.
    *   `metagha1oocyte`: Count of Metaphase I (MI) oocytes retrieved.
    *   `metaghaze2oocyte`: Count of Metaphase II (MII) oocytes retrieved (mature oocytes).
    *   `necrozeoocyte`: Count of necrotic (dead) oocytes retrieved.
    *   `lowQualityoocyte`: Count of oocytes deemed low quality upon retrieval.
    *   `Gvoocyte`: Count of Germinal Vesicle (GV) oocytes retrieved (immature).
    *   `macrooocyte`: Count of macro-oocytes (abnormally large) retrieved.
    *   `partogenesoocyte`: Count of spontaneously activated / parthenogenetic oocytes observed at denudation.
    *   `numberembrio`: Total count of embryos developed post-retrieval.
    *   `countspermogram`: Sperm count/concentration from baseline spermogram analysis.
    *   `motilityspermogram`: Percentage of motile sperm from baseline spermogram analysis (%).
    *   `morfologyspermogram`: Percentage of sperm with normal morphology from baseline spermogram analysis (%).
    *   `gradeembrio`: Quality grade assigned to the embryo(s) (Categorical: e.g., Grade 1, 2, 3).
    *   `Typecycle`: Type of treatment protocol used (Categorical: e.g., GnRH Agonist, GnRH Antagonist).
    *   `reasoninfertility`: Identified cause or source of infertility (Categorical: e.g., Female Factor, Male Factor, Both, Unexpected/Unexplained).
    *   `Typeofcunsumptiondrug`: Specific type of stimulation drug used (Categorical: e.g., Cinnal-f, Gonal-f, hMG).
    *   `typeoftrigger`: Type of drug used for final oocyte maturation trigger (Categorical: e.g., GnRH Agonist, hCG, Dual Trigger).
    *   `Typedrug`: Type of Drug Regimen/Protocol Detail (Categorical).
    *   `pregnancy`: History of previous pregnancy (Categorical: Positive/Negative).
    *   `mense`: Regularity of menstrual cycle at baseline (Categorical: Regular/Irregular).
    *   `Infertility`: Type of infertility at baseline (Categorical: Primary/Secondary).

*   **Target Variable (`OHSS`):** Must be present as a column. Encoded as **0 for Uncomplicated OHSS** (corresponding to original categories 'Painless' or 'Mild' OHSS) and **1 for Complicated OHSS** (corresponding to original categories 'Moderate' or 'Severe' OHSS).
*   **Data Preprocessing:** The script assumes the input data has already been handled for missing values (e.g., using imputation methods like those described in the paper - Random Forest for continuous, mean for categorical).

**For testing purposes:** You may want to create a small, synthetic, or anonymized dummy dataset following the expected structure and including *all* required column names (features + target) to run the script and verify its functionality.

## Usage

1.  **Modify Paths:** Open `tune_ohss_pipeline.py` and update the `train_data_path`, `test_data_path`, and `log_directory_base` variables to point to your data location and desired output directory.
2.  **Prepare Data:** Ensure your `train_data.xlsx` and `test_data.xlsx` files are in the correct location and contain all the required features (with exact names and described content) listed in the "Data" section above, along with the `OHSS` target variable (encoded 0/1).
3.  **Run the Script:** Execute the main script from your terminal within the activated virtual environment:
    ```bash
    python tune_ohss_pipeline.py
    ```
4.  **Process:** This script will initiate a Ray Tune hyperparameter optimization process, running a large number of trials (defined by `num_samples=15000`) to find the best combination of preprocessing steps, feature subsets, data augmentation techniques (SMOTE variants), models, and hyperparameters based on the `recall_mean` metric.
5.  **Output:**
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

This project is licensed under the MIT. See the `LICENSE` file for details.

## Contact

For questions regarding the research or code, please contact the author:
*   Arash Ziaee : `ziaeia961@mums.ac.ir`

---
