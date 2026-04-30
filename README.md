🏠 House Price Prediction using Machine Learning

📌 Overview

This project predicts house prices based on features like area, bedrooms, bathrooms, floors, age, and location score using a supervised machine learning approach.

---

⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

---
 🤖 Algorithm Used
* Random Forest Regressor (Supervised Learning)
* Linear Regression (for comparison)

🧠 How Prediction Works

The input data is first scaled using StandardScaler, and then passed into a Random Forest model.
The prediction is made by averaging outputs from multiple decision trees in the Random Forest model after scaling the input data.

---

📊 Features

* 250+ dataset records
* Data preprocessing using StandardScaler
* Model comparison (Linear Regression vs Random Forest)
* Model testing with separate file
* Interactive UI using Streamlit

---
 📁 Project Structure

house-price-ml/
│
├── app.py
├── train_model.py
├── test_model.py
│
├── data.csv
├── model.pkl
├── scaler.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

---
 ▶️ How to Run

 1. Install dependencies


pip install -r requirements.txt

 2. Train the model


python train_model.py

 3. Test the model

python test_model.py


4. Run the web app

streamlit run app.py


---
📈 Output

* MAE (Mean Absolute Error)
* R² Score
* Predicted house price

---

🎯 Conclusion

The Random Forest model provides better performance compared to Linear Regression by capturing complex relationships in the data and reducing prediction error.

---
