# 💳 Health Insurance Claim Fraud Detection
A machine learning–powered web application designed to detect fraudulent health insurance claim transactions.
This project integrates multiple ML models with a clean Streamlit interface, enabling real-time fraud prediction and model comparison.

🎯 Objectives :- 
- Predict fraudulent health insurance transactions using machine learning
- Build a user-friendly web interface for real-time predictions
- Compare model performance (Random Forest, Logistic Regression, XGBoost)
- Display model accuracy and prediction results clearly
- Analyze and preprocess transactional datasets for fraud detection

📂 Project Overview
This project focuses on detecting fraudulent activities in health insurance claim transactions using supervised machine learning.
The system allows users to:
- Input transaction details
- Select a model
- Instantly receive fraud/no-fraud predictions
- It uses a real-world dataset containing detailed transaction logs, balances, transaction types, and fraud indicators.

Key questions addressed include:
- Which transactions exhibit fraud patterns?
- How well do different ML models perform?
- What features strongly influence fraud decisions?

🧰 Tech Stack & Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Streamlit (Web application)
- Joblib / Pickle (Model saving & loading)
- Jupyter Notebook (Model training & EDA)

🧮 Machine Learning Models Used
- Random Forest Classifier
- Logistic Regression
- XGBoost Classifier
: Each model was trained, evaluated, and compared based on accuracy and prediction performance.

🧼 Data Overview
- Dataset Name: healthinsurance_fraud.csv
- The dataset includes these key fields:
- step — Transaction timestamp
- type — Transaction type (CASH-IN, CASH-OUT, TRANSFER, etc.)
- amount — Transaction amount
- nameOrig — Sender ID
- oldbalanceOrg — Sender’s initial balance
- newbalanceOrig — Sender’s balance after transaction
- nameDest — Receiver ID
- oldbalanceDest — Receiver’s initial balance
- newbalanceDest — Receiver’s balance after transaction
- isFraud — Fraud label (1 = Fraud, 0 = Genuine)
- isFlaggedFraud — System-flagged fraud indicator

📊 Workflow Summary
------------------------------------------
1️⃣ Loading & Understanding the Dataset
- Check structure
- Handle missing data
- Convert data types

2️⃣ Preprocessing
- Remove duplicates
- Encode categorical features
- Scale numerical features
- Feature engineering for fraud patterns

3️⃣ Model Training
- Train three ML models
- Hyperparameter tuning for Random Forest & XGBoost
- Save models using joblib

4️⃣ Web Application (Streamlit)
- Build user input form
- Provide model selection dropdown
- Display prediction + model accuracy

# 🚀 How to Run the Project
1️⃣ Clone the Repository
- git clone https://github.com/ayush13-0/health-insurance-claim-fraud-detection.git
- cd health-insurance-claim-fraud-detection

2️⃣ Create Conda Environment
- conda create -n fraud-detection python=3.9
- conda activate fraud-detection

3️⃣ Install Dependencies
- pip install -r requirements.txt

4️⃣ Ensure the Dataset is Present
- Place healthinsurance_fraud.csv in the project root.

5️⃣ Run the Streamlit App
- streamlit run fraud_detection_app.py
- Open the browser at:
👉 http://localhost:8501

📁 Project Structure
├── fraud_detection_app.py
├── fraud_model_training.ipynb
├── healthinsurance_fraud.csv
├── requirements.txt
└── README.md

👨‍💻 Developed By
-- Ayush
Data Science & Analytics | Machine Learning | Predictive Modeling
- 🔗 GitHub: https://github.com/ayush13-0
- 🔗 LinkedIn: https://www.linkedin.com/in/ayush130

📜 License
- This project is licensed under the **MIT License**.
