#Import Streamlit
import streamlit as st
#Other imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#Title of the dashboard
st.title("hii dear, I have started")

st.write("You can talk now")

#User uploading feature for input csv file
st.write("Upload your dataset (CSV file format)")
file = st.file_uploader("", type="csv")

read the csv file and display the dataframe
if file is not None:
    data = pd.read_csv(file)
    st.write("Preview of the uploaded dataset:")
    st.dataframe(data)

    target = st.selectbox('Select the target variable: ', 
    list(data.columns), index = list(data.columns).index(list(data.columns)[-1]))
    X = data.drop(columns=target)
    y = data[target]

# split the dataset into train and test and traina  logistic regrresison model
    st.write("Splitting the dataset into training and testing sets:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
    
    random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    st.write("Training a Logistic Regression Model:")
    model = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
    model.fit(X_train, y_train)

#Evaluate the model and print the accuracy score
    st.write("Evaluating the Model:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy: ", accuracy)

st.write("End of Training")
I

