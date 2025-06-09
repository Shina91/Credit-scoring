import streamlit as st
import pandas as pd
import pickle
import numpy as np

def load_model():
    with open("save_good_model.pkl", "rb") as file:
        data = pickle.load(file)
    return data

data = load_model()

rf_model = data["model"]
le_Occupation = data["le_Occupation"]
le_Payment_of_Min_Amount = data["le_Payment_of_Min_Amount"]
le_Payment_Behaviour = data["le_Payment_Behaviour"] 

credit_score_labels = {
    0: "Good",
    1: "Standard",
    2: "Bad"
}


# Function to show the prediction page
def show_predict_page():
    st.title("Credit Scoring Classification")
    st.write("""### Please provide the following information to predict the credit score classification""")

    occupations = (
        "Lawyer", "Teacher", "Mechanic", "Engineer", "Architect",
        "Scientist", "Entrepreneur", "Accountant", "Media_Manager",
        "Developer", "Journalist", "Doctor", "Musician", "Manager", "Writer",
    )

    payment_of_min_amount = ("Yes", "No", "NM")

    payment_behaviour = (
        "High_spent_Small_value_payments", "High_spent_Large_value_payments",
        "High_spent_Medium_value_payments", "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments", "Low_spent_Large_value_payments",
    )

    occupation = st.selectbox("Occupation", occupations)
    payment_of_min_amount = st.selectbox("Payment of Min Amount", payment_of_min_amount)    
    payment_behaviour = st.selectbox("Payment Behaviour", payment_behaviour)
    age = st.slider("Select Age", min_value=23, max_value=150, step=1)
    delay_from_due_date = st.slider("Select Delay from Due Date (days)", min_value=3, max_value=30, step=1)
    num_of_delayed_payment = st.number_input("Enter Number of Delayed Payments", value=7)
    outstanding_debt = st.number_input("Enter Outstanding Debt", value=809.98)
    credit_history_age = st.number_input("Enter Credit History Age (years)", value=22.1)
    total_emi_per_month = st.number_input("Enter Total EMI per month", value=49.57)
    monthly_balance = st.number_input("Enter Monthly Balance", value=312.49)
    annual_income = st.number_input("Annual Income", min_value=19114.12, max_value=1000000.00)

    ok = st.button("Classify Credit score")
    if ok:
        X = np.array([[age, occupation, annual_income, delay_from_due_date, num_of_delayed_payment,
                       outstanding_debt, credit_history_age, payment_of_min_amount,
                       total_emi_per_month, payment_behaviour, monthly_balance]])
        X[:,1] = le_Occupation.transform(X[:,1])
        X[:,7] = le_Payment_of_Min_Amount.transform(X[:,7])
        X[:,9] = le_Payment_Behaviour.transform(X[:,9])
        X = X.astype(float)

        credit_score = rf_model.predict(X)
        predicted_label = credit_score_labels[credit_score[0]]
       
        st.subheader(f"The predicted credit score classification is {predicted_label}")

    #st.write("""### Please provide further information to predict the Credit Score Classification""")



#data = pd.read_csv("Credit Score Dataset.csv")
#st.write(data)





#Classifier_name = st.sidebar.selectbox("Select Classifier", ("Random Forest", "SVM", "Naiyes Bayec"))
