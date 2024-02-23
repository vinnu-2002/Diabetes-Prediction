# Import required libraries
import numpy as np 
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Retrieve training and test data
# Data set: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
df = pd.read_csv("diabetes.csv")

cols_missing_vals = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']
df[cols_missing_vals] = df[cols_missing_vals].replace(to_replace=0, value=np.NaN)
df[cols_missing_vals] = df[cols_missing_vals].fillna(value=df.mean())

X = df.iloc[:,0:8]
y = df.iloc[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1992)

model = XGBClassifier(n_estimators=200, max_depth=25, learning_rate=0.1, subsample=0.5)
model.fit(X_train, y_train)

st.set_page_config(page_title='Diabet', page_icon='img/favicon.png', layout="wide", initial_sidebar_state="expanded", menu_items=None)
st.title('Diabetes Estimation')

st.subheader('Estimate your risk of developing Diabetes based on your biomarker values, using a trusted machine learning algorithm.')

st.sidebar.image('img/logo.png', use_column_width='auto', output_format='PNG')

st.sidebar.subheader('Insert your biomarker values here:')

def user_input():
    num_pregn = st.sidebar.slider('Number of Pregnancies',0,20,4)
    glucose = st.sidebar.slider('Glucose Level (mg/dL @ 2-Hour GTT)',0,240,121)
    blood_press = st.sidebar.slider('Diastolic Blood Pressure (mm Hg)',0,150,69)
    skin_thickn = st.sidebar.slider('Triceps Skin Fold Thickness (mm)',0,100,21)
    insulin = st.sidebar.slider('2-Hour Serum Insulin (mu U/ml)',0,850,80)
    bmi = st.sidebar.slider('Body Mass Index (kg/m^2)',0,70,32)
    diab_pedigr = st.sidebar.slider('Diabetes Pedrigree Value',0.0,2.5,0.5)
    age = st.sidebar.slider('Age (years)',0,85,33)

    user_input = {
        'Pregnancies': num_pregn,
        'Glucose': glucose,
        'BloodPressure': blood_press,
        'SkinThickness': skin_thickn,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diab_pedigr,
        'Age': age
    }

    user_data = pd.DataFrame(user_input, index=[0])
    return user_data

tab1, tab2 = st.tabs(['Overall Probability', 'Influencing Factors'])

with tab1:
    st.subheader('What is your risk of developing Diabetes?')
    user_outcome = model.predict_proba(user_input())
    st.metric(label='Probability', value=(st.write('{:.1%}'.format(user_outcome[0,1]))), delta_color='inverse',
    help='This is your calculated risk of developing Diabetes.')
    conclusion = ''
    if user_outcome[0,1] <= 0.3:
        conclusion = 'You **ARE NOT** at risk of  Diabetes.'
    else:
        conclusion = 'You **ARE** at risk of developing Diabetes. Seek medical consultation.'
    st.write(conclusion)

with tab2:
    st.subheader('Which factors contribute most to your risk of Diabetes?')

    # Plot most important features in desc order by 
    plot_feats = plot_importance(model, importance_type='gain', title=None, show_values=False, xlabel='Knowledge Gain', ylabel='Influencing Factor', color='#7E22CE', grid=False)
    st.pyplot(plot_feats.figure)