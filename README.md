# Heart Disease Risk Prediction

A machine-learning Streamlit application that analyzes clinical information and estimates the probability of heart disease. The application also provides interactive data exploration, model comparison, performance evaluation, and feature-importance visualizations.

## Live Application

The application is deployed on Streamlit Community Cloud:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-risk-prediction-fygpjne7bbgpvsqc8hcuxe.streamlit.app/)

[Launch the Heart Disease Risk Prediction application](https://heart-disease-risk-prediction-fygpjne7bbgpvsqc8hcuxe.streamlit.app/)

## Project Overview

Heart disease risk prediction is a binary classification problem. The application uses 13 clinical features from the Cleveland Heart Disease dataset to predict whether a patient is likely to have heart disease.

The project demonstrates a complete machine-learning workflow:

1. Load and clean the dataset.
2. Divide the data into training and testing sets.
3. Standardize features for models that require scaling.
4. Train and evaluate multiple classification models.
5. Compare model performance.
6. Generate individual risk predictions.
7. Explain important factors influencing predictions.

## Application Features

### Explore Data

- Review the dataset structure and summary statistics
- Examine the distribution of heart-disease outcomes
- Explore relationships between clinical features
- View the cleaned dataset in an interactive table

### Feature Importance

- Random Forest Gini feature importance
- Random Forest permutation importance
- Logistic Regression coefficients
- Visual interpretation of features associated with increased or decreased risk

### Model Comparison

- Compare five machine-learning classification models
- Review accuracy, precision, recall, F1 score, and AUC-ROC
- Examine confusion matrices and ROC curves
- Select a model from the application sidebar

### Risk Prediction

- Enter patient information using interactive controls
- Generate a binary risk classification
- Display the estimated probability of heart disease
- Review the individual clinical values used by the model

## Machine-Learning Models

The application trains and compares:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine
- K-Nearest Neighbours

Logistic Regression, Support Vector Machine, and K-Nearest Neighbours use standardized input features. Tree-based models use the original feature values.

## Model Evaluation

The dataset is divided into training and held-out testing sets. Model predictions on the testing set are evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- AUC-ROC
- Confusion matrix
- Classification report
- Cross-validation score

The test-set percentage and random seed can be adjusted from the application sidebar.

## Dataset

The application uses the processed Cleveland Heart Disease dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data).

The dataset contains approximately 303 patient records and 13 clinical predictor variables:

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Biological sex |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar above 120 mg/dL |
| `restecg` | Resting electrocardiogram result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels identified by fluoroscopy |
| `thal` | Thalassemia test result |

The original outcome is converted into a binary target:

- `0`: No heart disease detected
- `1`: Heart disease detected

Missing observations are removed before model training. If the live UCI dataset cannot be downloaded, the application uses a synthetic fallback dataset so the interface can still run.

## Technologies Used

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Project Structure

```text
heart-disease-risk-prediction/
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Download or clone the repository

Open the repository folder in Visual Studio Code.

### 2. Create a virtual environment

On Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

Alternatively, create a `requirements.txt` file containing:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Then install it with:

```bash
python -m pip install -r requirements.txt
```

## Running the Application

From the VS Code terminal, run:

```bash
python -m streamlit run streamlit_app.py
```

Streamlit should open the application automatically. If it does not, open:

```text
http://localhost:8501
```

## Streamlit Community Cloud Deployment

Live deployment:

<https://heart-disease-risk-prediction-fygpjne7bbgpvsqc8hcuxe.streamlit.app/>

To deploy another version of the application:

1. Upload `streamlit_app.py`, `requirements.txt`, and `README.md` to GitHub.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Select **Create app**.
4. Choose the GitHub repository.
5. Set the main file path to `streamlit_app.py`.
6. Deploy the application.

## Example Workflow

1. Open **Explore Data** to review the dataset.
2. Open **Feature Importance** to examine influential clinical variables.
3. Open **Model Comparison** to compare classification results.
4. Select the preferred model from the sidebar.
5. Open **Predict** and enter patient information.
6. Select **Run Prediction** to generate the estimated risk.

## Limitations

- The dataset is relatively small and represents a specific patient population.
- Model performance may change when applied to patients from other populations.
- Predictions depend on the quality and accuracy of the entered data.
- Feature importance identifies statistical patterns and does not prove causation.
- The application has not been validated as a clinical decision-support system.

## Medical Disclaimer

This application was created for educational and demonstration purposes only. It is not a medical device and must not be used to diagnose heart disease, replace professional clinical judgment, or guide treatment decisions. Individuals concerned about their cardiovascular health should consult a qualified healthcare professional.

## Future Improvements

- Validate the models using a larger and more diverse dataset
- Add hyperparameter optimization
- Add external validation data
- Save trained models for faster deployment
- Add SHAP explanations for individual predictions
- Add secure prediction-history export
- Improve accessibility and mobile responsiveness

## Author

Muluwerk Derebe  
Business Intelligence Student

## Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for the Cleveland Heart Disease dataset
- [Streamlit](https://streamlit.io/) for the interactive application framework
- [scikit-learn](https://scikit-learn.org/) for machine-learning tools
