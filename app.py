
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# App title
st.title("Diabetes Progression Regression")

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R-squared Score:** {r2:.2f}")

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# True vs Predicted
axs[0].scatter(y_test, y_pred, color='blue', alpha=0.5)
axs[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'k--',
    lw=2
)
axs[0].set_xlabel("True Values")
axs[0].set_ylabel("Predicted Values")
axs[0].set_title("True vs Predicted Values")

# BMI feature vs Predicted values (BMI = feature index 2)
axs[1].scatter(X_test[:, 2], y_pred, color='green', alpha=0.7)
axs[1].set_xlabel("BMI Feature")
axs[1].set_ylabel("Predicted Values")
axs[1].set_title("BMI vs Predicted Diabetes Progression")

# Display plot in Streamlit
st.pyplot(fig)
