# 🏥 PredMed Linear Regression

A custom-built multivariate linear regression model to **predict medical insurance charges** based on user demographics and health information.  
No `scikit-learn`, just raw `NumPy` math and custom preprocessing 🚀

---

## 📌 Features

- ✅ Fully implemented linear regression from scratch
- 🔁 Gradient descent optimizer
- 📊 Mean squared error (MSE) tracking and plotting
- ⚙️ Feature scaling (standardization)
- 🧠 One-hot encoding for categorical data
- 🧪 Predict charges for new user input

---

## 🧮 Inputs Used

This model predicts `charges` based on:

- `age`: Age of the individual
- `bmi`: Body Mass Index
- `children`: Number of children/dependents
- `smoker`: Whether the person smokes (1 for yes, 0 for no)
- `sex`: Converted to binary (`male`: 1 or 0)
- `region`: One-hot encoded for `southwest`, `northwest`, `northeast`
- `bias`: Constant added internally for the intercept term

---

## 🚀 Getting Started

```bash
git clone https://github.com/ruinedm/predmed_lr.git
cd predmed_lr
pip install -r requirements.txt
python3 main.py
```
This will train the model and compare it against sickit-learn's LinearRegression, feel free to test it however you like :)

This project marks the beginning of my machine learning journey! I'm still learning and experimenting, so any feedback, comments, or pull requests are more than welcome and appreciated. Feel free to contribute or suggest improvements to help me grow. Thanks for checking it out! 🙌


