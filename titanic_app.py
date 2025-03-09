import pandas as pd
import pickle
import streamlit as st

# Load the pre-trained model
with open('titanic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Title of the app
st.title('Titanic Survival Prediction')

# Input features
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 30)
sibsp = st.slider('SibSp', 0, 10, 0)
fare = st.slider('Fare', 0, 500, 10)
embarked = st.selectbox('Embarked', ['Q', 'S', 'C'])

# Convert categorical variables to numerical
sex = 0 if sex == 'male' else 1
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

# Create DataFrame for prediction
input_data = pd.DataFrame([[pclass, sex, age, sibsp, fare, embarked_Q, embarked_S]],
                          columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked_Q', 'Embarked_S'])

# Predict survival
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write('Survival Prediction: ', 'Survived' if prediction[0] == 1 else 'Did not survive')
