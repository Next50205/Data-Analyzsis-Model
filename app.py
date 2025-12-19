import streamlit as st
import pandas as pd

st.set_page_config(
    page_title= "Data Analyzsis Model",
    page_icon= ":crown:",
    layout="centered"
)

st.title("This is Sustainable Waste Management Dataset 2024", text_alignment='center')
df = pd.read_csv('sustainable_waste_management_dataset_2024.csv')

if st.toggle("Open the Dataframe"): 
    st.dataframe(df)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng as rng    
CSV_FILE = "sustainable_waste_management_dataset_2024.csv"
df = pd.read_csv(CSV_FILE)
df.head()    

selected_features = ['recyclable_kg', 'collection_capacity_kg', 'temp_c', 'rain_mm']
X = df[selected_features]
y = df['waste_kg']

df_combined = pd.concat([X,y], axis = 1)
df_combined.dropna(inplace=True)

X = df[selected_features]
y = df['waste_kg']

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], '--',color = 'red' , lw=2,label='Perfect Prediction Line')
ax.set_xlabel("Actual Waste (kg)")
ax.set_ylabel("Predicted Waste (kg)")
ax.set_title("Actual vs. Predicted Waste")
ax.legend()
ax.grid(True)
st.pyplot(fig)



