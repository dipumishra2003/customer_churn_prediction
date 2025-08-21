📊 Customer Churn Prediction

This repository contains a machine learning project for predicting customer churn. Churn prediction is a critical task for businesses, as it helps identify customers who are likely to leave a service or product. By predicting churn in advance, companies can take proactive measures such as offering discounts, improving customer service, or launching targeted campaigns to retain valuable customers.

The project applies data preprocessing, handling class imbalance with SMOTE, and multiple classification models (Decision Tree, Random Forest, and XGBoost). It evaluates models using accuracy, confusion matrix, and classification reports, and saves the best-performing model for future use with Pickle.

🚀 Features

Data preprocessing and encoding categorical variables

Handling class imbalance using SMOTE

Model training with:

Decision Tree

Random Forest

XGBoost

Model evaluation with Accuracy, Confusion Matrix, and Classification Report

Model persistence using Pickle

📂 Project Structure
├── data/                  # Dataset files (not included, add your own)
├── models/                # Saved trained models (.pkl)
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── README.md              # Project documentation
└── requirements.txt       # List of dependencies

🛠️ Dependencies

The project requires the following Python libraries:

numpy
pandas
matplotlib
seaborn
scikit-learn
imblearn
xgboost
pickle


Install dependencies with:

pip install -r requirements.txt

⚙️ Usage

Clone this repository:

git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction


Add your dataset inside the data/ folder.

Run the training script:

python src/train_model.py


Evaluate the trained model:

python src/evaluate_model.py


Use the saved model (.pkl file) for predictions:

import pickle
model = pickle.load(open("models/churn_model.pkl", "rb"))
prediction = model.predict([[...]])
print("Churn Prediction:", prediction)

📈 Results

Models compared using Accuracy, Precision, Recall, and F1-score

Confusion matrices and classification reports for performance insights

🔮 Future Improvements

Add more advanced algorithms (LightGBM, CatBoost)

Perform hyperparameter tuning

Deploy with Flask / FastAPI as an API

Build a dashboard with Streamlit

🤝 Contributing

Contributions are welcome! Fork the repo and open a pull request 🚀

📜 License

This project is licensed under the MIT License.
