import pandas as pd
from model import PredMedLinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- Custom model ---
custom_model = PredMedLinearRegression("insurance.csv")
custom_model.train()
custom_model.plot_mse("mse_plot.png")
custom_mse = custom_model.get_mse()

test_data = {
    "age": 35,
    "bmi": 27.5,
    "children": 2,
    "smoker": 1,
    "male": 1,
    "region": "southwest"
}

custom_prediction = custom_model.predict(test_data)
print(f"[Custom] Predicted charge: {custom_prediction:.2f}")
print(f"[Custom] Final MSE: {custom_mse:.2f}")

# --- Sklearn model using full data ---
df = pd.read_csv("insurance.csv")
df["smoker"] = (df["smoker"] == "yes").astype(int)
df["male"] = (df["sex"] == "male").astype(int)
df["southwest"] = (df["region"] == "southwest").astype(int)
df["northwest"] = (df["region"] == "northwest").astype(int)
df["northeast"] = (df["region"] == "northeast").astype(int)
df.drop(columns=["region", "sex"], inplace=True)

X = df.drop("charges", axis=1)
y = df["charges"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


sklearn_model = LinearRegression()
sklearn_model.fit(X_scaled, y)
y_pred = sklearn_model.predict(X_scaled)
sklearn_mse = mean_squared_error(y, y_pred)


test_input = pd.DataFrame([test_data])
test_input["southwest"] = (test_input["region"] == "southwest").astype(int)
test_input["northwest"] = (test_input["region"] == "northwest").astype(int)
test_input["northeast"] = (test_input["region"] == "northeast").astype(int)
test_input["male"] = (test_input["male"] == 1).astype(int)
test_input["smoker"] = (test_input["smoker"] == 1).astype(int)
test_input = test_input.drop(columns=["region"])
test_input_scaled = scaler.transform(test_input)

sklearn_prediction = sklearn_model.predict(test_input_scaled)[0]

print(f"[Sklearn] Predicted charge: {sklearn_prediction:.2f}")
print(f"[Sklearn] Final MSE: {sklearn_mse:.2f}")
print(f"MSE difference: {abs(custom_mse - sklearn_mse):.2f}")
