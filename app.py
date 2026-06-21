
import streamlit as st
import joblib
import pandas as pd

# Load model dan scaler
bundle = joblib.load("model_bundle.pkl")

model = bundle["logreg"]
scaler = bundle["scaler_lr"]

# Konfigurasi halaman
st.set_page_config(
    page_title="Cardiovascular Risk Assessment",
    layout="centered"
)

# Header
st.title("Cardiovascular Risk Assessment")

st.caption(
    "Estimasi kelompok risiko kardiovaskular berdasarkan usia, BMI, "
    "kolesterol, dan tekanan darah sistolik."
)

st.divider()

# Input Data
st.subheader("Data Pasien")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Usia (tahun)",
            min_value=1,
            max_value=120,
            value=30
        )

        chol = st.number_input(
            "Total Cholesterol (mg/dL)",
            min_value=50.0,
            max_value=500.0,
            value=180.0
        )

    with col2:
        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0
        )

        bp = st.number_input(
            "Systolic BP (mmHg)",
            min_value=50.0,
            max_value=250.0,
            value=120.0
        )

    submit = st.form_submit_button(
        "Jalankan Asesmen",
        use_container_width=True
    )

# Hasil Prediksi
if submit:

    input_data = pd.DataFrame(
        [[age, bmi, chol, bp]],
        columns=[
            "Age",
            "BMI",
            "Total Cholesterol (mg/dL)",
            "Systolic BP"
        ]
    )

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    st.divider()
    st.subheader("Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Cluster",
            prediction
        )

    with col2:
        st.metric(
            "Kategori Risiko",
            "Rendah" if prediction == 0 else "Tinggi"
        )

    summary = pd.DataFrame({
        "Parameter": [
            "Usia",
            "BMI",
            "Kolesterol",
            "Systolic BP"
        ],
        "Nilai": [
            f"{age} tahun",
            f"{bmi:.1f}",
            f"{chol:.0f} mg/dL",
            f"{bp:.0f} mmHg"
        ]
    })

    with st.container(border=True):
        st.subheader("Ringkasan Data")
        st.table(summary)

    if prediction == 0:
        st.success(
            "Pasien termasuk ke dalam kelompok Risiko Rendah."
        )
    else:
        st.error(
            "Pasien termasuk ke dalam kelompok Risiko Tinggi."
        )

    st.info(
        "Hasil ini merupakan keluaran model Logistic Regression "
        "dan digunakan sebagai alat bantu analisis, bukan diagnosis medis."
    )

else:
    st.info(
        "Masukkan data pasien lalu klik 'Jalankan Asesmen' untuk melihat hasil prediksi."
    )
