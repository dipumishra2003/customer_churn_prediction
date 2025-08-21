# 📊 Customer Churn Prediction  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ScikitLearn%2C%20XGBoost-orange)  
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  

Customer churn prediction is a **machine learning project** that helps businesses identify customers who are likely to discontinue a service. By predicting churn, companies can take **proactive steps** such as offering discounts, improving services, or launching personalized campaigns to retain customers.  

This project applies **data preprocessing, class imbalance handling with SMOTE, and classification models (Decision Tree, Random Forest, XGBoost)**. It evaluates models with metrics like **Accuracy, Confusion Matrix, and Classification Report** and saves the trained model for future use with **Pickle**.  

---

## 🚀 Features
✅ Data preprocessing and encoding categorical variables  
✅ Handling imbalance using **SMOTE**  
✅ Models implemented: **Decision Tree, Random Forest, XGBoost**  
✅ Model evaluation with **accuracy, precision, recall, F1-score**  
✅ Save and load trained model with **Pickle**  

---

## 📂 Project Structure

```
├── data/ # Dataset files (add your dataset here)
├── models/ # Saved trained models (.pkl)
├── notebooks/ # Jupyter notebooks for exploration
├── src/ # Source code
│ ├── data_preprocessing.py
│ ├── train_model.py
│ └── evaluate_model.py
├── README.md # Project documentation
└── requirements.txt # List of dependencies
```


---

## 🛠️ Installation & Dependencies

Clone the repository:

```
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

Install dependencies:

pip install -r requirements.txt
```

# Requirements:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

# ⚙️ Usage

1️⃣ Train the Model
python src/train_model.py

2️⃣ Evaluate the Model
python src/evaluate_model.py

3️⃣ Load & Predict
import pickle

# Load saved model
model = pickle.load(open("models/churn_model.pkl", "rb"))

# Example prediction
prediction = model.predict([[...]])
print("Churn Prediction:", prediction)

# 📈 Results

Models evaluated with Accuracy, Precision, Recall, and F1-score

Confusion matrices and classification reports generated for deeper insights

# 🔮 Future Improvements

Add more advanced algorithms (LightGBM, CatBoost)

Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

Deploy as an API using Flask / FastAPI

Build interactive dashboards with Streamlit / Dash

# 🤝 Contributing

Contributions are welcome!

Fork the repository

Create your feature branch (git checkout -b feature/YourFeature)

Commit your changes (git commit -m 'Add feature')

Push to the branch (git push origin feature/YourFeature)

Open a Pull Request

# 📜 License

This project is licensed under the MIT License.
