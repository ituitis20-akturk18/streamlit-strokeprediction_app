import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

model = joblib.load("pipeline_stroke.pkl")

def stroke_data_prep(df):
    df.columns = [col.upper() for col in df.columns]

    df['NEW_AVG_GLUCOSE_LEVEL'] = pd.cut(x=df['AVG_GLUCOSE_LEVEL'], bins=[0, 85, 165, 230],
                                         labels=["low", "normal", "high"])

    df.loc[df["AGE"] < 13, "NEW_AGE_CAT"] = "children"
    df.loc[(df["AGE"] >= 13) & (df["AGE"] <= 18), "NEW_AGE_CAT"] = "teens"
    df.loc[(df["AGE"] > 18) & (df["AGE"] <= 35), "NEW_AGE_CAT"] = "adult"
    df.loc[(df['AGE'] >= 35) & (df['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
    df.loc[(df['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

    df['NEW_BMI_RANGE'] = pd.cut(x=df['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                                 labels=["underweight", "healthy", "overweight", "obese"])

    label_encode = LabelEncoder()

    df["NEW_AVG_GLUCOSE_LEVEL"] = label_encode.fit_transform(df["NEW_AVG_GLUCOSE_LEVEL"])
    df["NEW_AGE_CAT"] = label_encode.fit_transform(df["NEW_AGE_CAT"])
    df["NEW_BMI_RANGE"] = label_encode.fit_transform(df["NEW_BMI_RANGE"])

    df = pd.get_dummies(df, columns=["WORK_TYPE", "SMOKING_STATUS"], drop_first=True)

    for col in ["AGE", "AVG_GLUCOSE_LEVEL", "BMI"]:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])

    return df

# Verileri içeren küçük bir DataFrame oluşturur
def create_input_dataframe(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type,
                           avg_glucose_level, bmi, smoking_status):
    data = {
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    }
    input_df = pd.DataFrame(data)
    return input_df


def make_prediction(scaled_df):

    prediction = model.predict(scaled_df)
    return prediction

def main():
    st.title("Stroke Risk Prediction App")

    gender = st.selectbox("Gender:", ["Male", "Female"])
    age = st.number_input("Age:", min_value=1, max_value=100)
    hypertension = st.checkbox("Hypertension")
    heart_disease = st.checkbox("Heart Disease")
    ever_married = st.radio("Ever Married:", ["Yes", "No"])
    work_type = st.selectbox("Work Type:", ["Private", "Self-employed", "Govt_job", "Never_worked"])
    residence_type = st.radio("Residence Type:", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level:", min_value=0.0, max_value=500.0)
    bmi = st.number_input("Body Mass Index (BMI):", min_value=0.0, max_value=100.0)
    smoking_status = st.selectbox("Smoking Status:",
                                  ["Unknown", "Never smoked", "Formerly smoked", "Currently smoking"])

    input_df = create_input_dataframe(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type,
                                      avg_glucose_level, bmi, smoking_status)

    scaled_df = stroke_data_prep(input_df)

    if st.button("Predict"):
        prediction = make_prediction(scaled_df)
        st.write("Prediction Result:", prediction)


if __name__ == "__main__":
    main()
