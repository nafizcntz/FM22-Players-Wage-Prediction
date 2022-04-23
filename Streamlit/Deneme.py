import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
import joblib
import streamlit.components.v1 as components

st.set_page_config(layout="wide")


df = pd.read_csv(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\Final Projesi\Turkcell GY DS Bootcamp Projesi\Streamlit\Model_deployment.csv")
df_2 = pd.read_csv(r"/app/fm22-players-wage-prediction/Streamlit/Model_deployment_2.csv")
df_X_test = pd.read_csv(r"/app/fm22-players-wage-prediction/Streamlit/Model_X_test.csv")

model = joblib.load(r'C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\Final Projesi\Turkcell GY DS Bootcamp Projesi\Streamlit\voting_clf_2.pkl')
