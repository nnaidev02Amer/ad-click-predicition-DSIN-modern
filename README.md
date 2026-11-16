# DSIN – Advertisement Click-Through Rate Prediction

### Deep Session Interest Network for Online Advertisement Click Prediction

DSIN is a deep learning-based project that predicts the probability of a user clicking an online advertisement based on user information, previously viewed products, browsing sessions and advertisement details.

The project implements a user-friendly Streamlit dashboard that allows users to enter information, select browsing history, provide advertisement details and obtain a predicted click probability using a trained DSIN model.

**Project Type:** Final-Year Engineering Project  
**Domain:** Machine Learning / Deep Learning / Artificial Intelligence  
**Model:** Deep Session Interest Network (DSIN)  
**Framework:** TensorFlow / Keras  
**Interface:** Streamlit  
**Current Dataset Mode:** Synthetic Demonstration Data

---

## 1. Project Overview

Online advertising is an important part of e-commerce platforms. Businesses display advertisements to users based on their interests and browsing behaviour.

However, not every user clicks every advertisement.

Click-Through Rate (CTR) prediction helps estimate the likelihood that a user will click an advertisement when it is displayed.

This project uses the Deep Session Interest Network (DSIN), a deep learning architecture designed to model user interests from browsing sessions.

The system analyses:

- User profile information.
- User browsing history.
- Previously viewed product categories.
- Product brands.
- Advertisement information.
- Advertisement price and related features.

The DSIN model processes these inputs and produces a predicted advertisement click probability.

**Important:** The current dashboard uses synthetic demonstration data. The model has not yet been validated using the original Taobao dataset. Its demonstration results must not be interpreted as real-world advertising performance.

---

## 2. Project Objectives

The main objectives are:

1. Understand user browsing behaviour.
2. Represent browsing history as multiple sessions.
3. Extract user interests from browsing sessions.
4. Apply a Deep Session Interest Network architecture.
5. Predict the probability of clicking an advertisement.
6. Build an interactive and understandable dashboard.
7. Demonstrate the prediction process using sample inputs.

---

## 3. What Is CTR?

CTR stands for Click-Through Rate.

It measures the percentage of advertisement impressions that result in clicks.

Formula:

CTR = (Number of Clicks / Number of Impressions) × 100

Example:

- Total advertisement impressions: 1,000
- Total clicks: 73

CTR = (73 / 1000) × 100 = 7.3%

The DSIN project estimates the probability of a click for a particular user and advertisement.

A predicted probability is an estimate, not a guarantee that the user will click.

---

## 4. What Is DSIN?

DSIN stands for Deep Session Interest Network.

It is a deep learning architecture designed to model user interests from multiple browsing sessions.

Instead of treating all previously viewed products as one flat sequence, DSIN considers the user's browsing sessions.

For example:

**Most Recent Visit**

- Smartphones
- Earphones
- Smart Watches

**Previous Visit**

- Laptops
- Tablets

**Earlier Visit**

- Accessories
- Bags

These sessions can help represent different user interests.

### Main DSIN Components

**1. Session Division**

Separates browsing behaviour into individual sessions.

**2. Session Interest Extraction**

Extracts meaningful interests from the products viewed during each session.

**3. Multi-Head Self-Attention**

Helps the model learn relationships between items within a session.

**4. Bias Encoding**

Provides additional information about item positions and session structure.

**5. Bi-LSTM / Session Interaction**

Models relationships and sequential information across sessions.

**6. Target Attention**

Helps identify user interests relevant to the advertisement being evaluated.

**7. DNN Prediction Layer**

Processes the learned representations and generates a click probability.

---

## 5. System Workflow

The project follows this workflow:

1. Collect user profile information.
2. Select previously viewed products.
3. Organize browsing history into sessions.
4. Enter advertisement details.
5. Convert user-friendly selections into model-compatible numerical features.
6. Pass the processed features to the trained DSIN model.
7. Generate a click probability.
8. Display the prediction in the Streamlit dashboard.

### Workflow Diagram

```text
        User Information
               |
               v
       Browsing History
               |
               v
       Session Processing
               |
               v
     Advertisement Details
               |
               v
       Feature Processing
               |
               v
        DSIN Deep Model
               |
               v
       Click Probability
               |
               v
       Streamlit Results
```

---

## 6. Dashboard Features

The application contains four main sections.

### 1. Home

- Project introduction.
- Problem statement.
- Project objectives.
- System workflow.
- Technology information.

### 2. Predict Advertisement Click

Allows users to provide:

**User Information**

- Age.
- Gender.

**Products Previously Viewed**

- Most Recent Visit.
- Previous Visit.
- Earlier Visit.
- Multiple product selections per visit.
- Maximum 10 products per session.

**Advertisement Details**

- Advertisement/product name.
- Product category.
- Product brand.
- Product price.

**Advanced Details**

- Campaign ID.
- Customer ID.
- Advertisement placement.

The dashboard converts supported selections into the numerical inputs required by the DSIN model.

### 3. Prediction Results

Displays:

- Predicted click probability.
- Comparison with the average click rate of the training data.
- Probability visualization.
- User information summary.
- Advertisement summary.
- Browsing history summary.
- Explanation of the predicted result.

### 4. About DSIN

Explains:

- Click-Through Rate.
- DSIN architecture.
- Session-based interests.
- Attention mechanisms.
- Prediction workflow.
- Technologies used.

---

## 7. Technology Stack

| Technology      | Purpose                                     |
| --------------- | ------------------------------------------- |
| Python 3.9      | Programming language                        |
| TensorFlow 2.15 | Deep learning framework                     |
| Keras 2.15      | Neural network API                          |
| DeepCTR 0.9.4   | DSIN model implementation                   |
| NumPy           | Numerical operations                        |
| Pandas          | Dataset processing                          |
| Scikit-learn    | Data preprocessing and evaluation utilities |
| Streamlit       | Interactive dashboard                       |
| Plotly          | Data visualization                          |
| Joblib          | Model and data utilities                    |
| Pytest          | Automated testing                           |

---

## 8. Project Structure

The important project files are organized as follows:

```text
DSIN-modern/
│
├── app.py
│
├── dsin/
│   ├── labels.py
│   ├── model.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── history.py
│
├── dashboard/
│   ├── page_home.py
│   ├── page_predict.py
│   ├── page_results.py
│   └── page_about.py
│
├── tests/
│   └── test_dashboard.py
│
├── raw_data/
│
├── scripts/
│
├── models/
│
├── requirements-modern.txt
├── README.md
├── README-MODERN.md
└── setup_windows.ps1
```

Note: Some directories and files may differ depending on the current project version. Refer to the actual repository for the complete file listing.

---

## 9. System Requirements

Recommended environment:

- Windows 10 or Windows 11.
- Python 3.9.
- 8 GB RAM recommended for comfortable development.
- Sufficient free disk space for dependencies, generated data and model files.
- Internet connection for initial package installation.

The project was modernized to use TensorFlow 2.15 rather than the legacy TensorFlow 1.x environment used by the original implementation.

---

## 10. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/nnaidev02Amer/ad-click-predicition-DSIN-modern.git
```

### Step 2: Navigate to the Project

```bash
cd ad-click-predicition-DSIN-modern
```

### Step 3: Create a Virtual Environment

Using Python 3.9:

```bash
py -3.9 -m venv venv
```

If the Python launcher cannot find Python 3.9, use the full path to your Python 3.9 executable.

### Step 4: Activate the Virtual Environment

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can continue using the virtual environment's Python executable directly.

### Step 5: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 6: Install Dependencies

```bash
pip install -r requirements-modern.txt
```

If the project has a Windows setup script, you can also inspect and use:

```powershell
.\setup_windows.ps1
```

Use the setup script only if it matches the current environment configuration.

---

## 11. Running the Application

Start the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

### Run on Another Port

If port 8501 is already being used:

```bash
python -m streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

### Stop the Application

Press:

```text
Ctrl + C
```

in the terminal running Streamlit.

---

## 12. How to Use the Application

### Step 1: Open Home

Understand the project and its objective.

### Step 2: Open About DSIN

Review the model architecture and session-based interest extraction.

### Step 3: Open Predict Advertisement Click

Enter the user's age and gender.

### Step 4: Select Browsing History

Choose products viewed during the most recent, previous and earlier visits.

### Step 5: Enter Advertisement Details

Select the advertisement, product category, brand and price.

### Step 6: Predict

Click:

**Predict Click Probability**

### Step 7: View Results

Review the predicted percentage and its explanation.

The prediction represents an estimate generated by the trained model.

---

## 13. Dataset Information

### Current Demonstration Dataset

The current working demonstration uses synthetic data.

Synthetic data is artificially generated for testing and demonstrating the application.

It allows the project to demonstrate:

- Data processing.
- Session construction.
- Feature mapping.
- Model execution.
- Prediction generation.
- Dashboard functionality.

The illustrative Indian product names and brands shown in the interface are not actual Taobao product records.

### Original Dataset

The original DSIN research implementation uses the Taobao dataset.

The expected source files in the original data workflow include:

```text
raw_sample.csv
ad_feature.csv
user_profile.csv
raw_behavior_log.csv
```

The actual filenames and required formats must be checked against the preprocessing implementation.

Dataset source:

https://tianchi.aliyun.com/dataset/dataDetail?dataId=56

The current project has not been validated on the complete original dataset.

---

## 14. Model Training and Evaluation

The project includes scripts for preprocessing, training and evaluation.

Typical commands in the modernized project:

```bash
python run_preprocessing.py --mode demo --make-demo
```

Train the synthetic demonstration model:

```bash
python train_model.py --mode demo --epochs 6 --batch-size 512
```

Evaluate the model:

```bash
python evaluate_model.py
```

Always check the script's available arguments and current project configuration before running these commands.

### Important Model Limitation

The current synthetic demonstration model has an AUC close to 0.50.

An AUC around 0.50 indicates approximately chance-level ability to distinguish clicked from non-clicked examples.

Therefore:

- The current output demonstrates the prediction workflow.
- It does not establish reliable real-world advertising performance.
- Synthetic evaluation metrics must not be presented as Taobao results.

---

## 15. Running Tests

Run the complete test suite:

```bash
python -m pytest tests -q
```

The previous development report recorded 32 passing tests. Test counts may change as the project develops.

The tests cover areas such as:

- Dashboard pages.
- Input validation.
- Feature mappings.
- Browsing sessions.
- Prediction workflow.
- Model input compatibility.
- Prediction results.

---

## 16. Important Limitations

1. The current demonstration uses synthetic data.
2. The model has not been validated using the original Taobao dataset.
3. Synthetic AUC is close to 0.50.
4. Demonstration product and brand labels are illustrative.
5. Predicted click probability is an estimate, not a guaranteed click.
6. Real-world performance requires appropriate real data, training and evaluation.

---

## 17. Future Enhancements

Possible future improvements include:

- Training with the original Taobao dataset.
- Evaluating model performance on real data.
- Improving user-interest representations.
- Adding more realistic advertisement examples.
- Improving prediction explanations.
- Comparing predicted performance against a properly defined baseline.

These are future possibilities and are not claimed as completed features.

---

## 18. Research Reference

**Original Research Paper**

Feng et al. (2019).

Deep Session Interest Network for Click-Through Rate Prediction.

International Joint Conference on Artificial Intelligence (IJCAI 2019).

Research paper:

https://arxiv.org/abs/1905.06482

Original implementation:

https://github.com/shenweichen/DSIN

---

## 19. Author and Project Information

**Project:** Advertisement Click-Through Rate Prediction Using DSIN

**Project Category:** Final-Year Engineering Project

**Domain:** Artificial Intelligence and Deep Learning

**Implementation:** Python, TensorFlow, Keras, DeepCTR and Streamlit

**Repository:**

https://github.com/nnaidev02Amer/ad-click-predicition-DSIN-modern

---

## 20. License and Usage

This repository is intended for educational and academic demonstration.

Please review the licensing terms of the original DSIN implementation and its dependencies before redistributing the project.

---

**Final Note**

This project demonstrates how a Deep Session Interest Network can be used to estimate advertisement click probability from user profiles, browsing sessions and advertisement information.

The current version is an educational demonstration using synthetic data. Real-world performance has not yet been established.
