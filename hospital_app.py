# ==========================================================
# SMART HOSPITAL PATIENT NAVIGATOR
# Future Classroom - Machine Learning Project
# ==========================================================

# Import library yang diperlukan
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Smart Hospital Patient Navigator",
    page_icon="🏥",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# Digunakan untuk mempercantik tampilan aplikasi
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none; }
.stDeployButton { display: none; }
footer { visibility: hidden; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}

div[data-testid="stForm"] {
    border: none;
    padding: 0;
}

div.stButton > button {
    background: linear-gradient(135deg, #1a56db, #1e429f) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(26,86,219,0.35) !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #1e429f, #1a56db) !important;
}

div[data-testid="stCheckbox"] label {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #374151 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MACHINE LEARNING MODEL
# Model yang sudah dilatih sebelumnya disimpan
# dalam file hospital_model.pkl
# ==========================================================

@st.cache_resource
def load_model():
    # Membuka file model
    with open('hospital_model.pkl', 'rb') as f:
        return pickle.load(f)

bundle = load_model()

# Mengambil seluruh komponen yang tersimpan di model
model = bundle['model']
scaler = bundle['scaler']
features = bundle['features']
cols_to_scale = bundle['cols_to_scale']
dept_map_inv = bundle['dept_map_inv']
gender_map = bundle['gender_map']
temp_map = bundle['temp_map']
hr_map = bundle['hr_map']
dur_map = bundle['dur_map']
cc_map = bundle['cc_map']

# ==========================================================
# INFORMASI DEPARTEMEN RUMAH SAKIT
# Akan ditampilkan setelah AI memberikan rekomendasi
# ==========================================================

DEPT_INFO = {

    'Respiratory Medicine': {
        'icon':'🫁',
        'color':'#0284c7',
        'bg':'#e0f2fe',
        'border':'#7dd3fc',
        'desc':'Specialises in conditions affecting the lungs and airways.',
        'next':[
            'Visit Level 2, Wing B',
            'Estimated wait: 15–25 min',
            'Please wear a mask'
        ]
    },

    'Cardiology': {
        'icon':'❤️',
        'color':'#dc2626',
        'bg':'#fee2e2',
        'border':'#fca5a5',
        'desc':'Specialises in heart and cardiovascular conditions.',
        'next':[
            'Visit Level 3, Wing A',
            'Estimated wait: 20–30 min',
            'Bring any previous ECG reports'
        ]
    },

    'Gastroenterology': {
        'icon':'🫃',
        'color':'#d97706',
        'bg':'#fef3c7',
        'border':'#fcd34d',
        'desc':'Specialises in digestive system and abdominal conditions.',
        'next':[
            'Visit Level 1, Wing C',
            'Estimated wait: 10–20 min',
            'Avoid eating before consultation'
        ]
    },

    'Neurology': {
        'icon':'🧠',
        'color':'#7c3aed',
        'bg':'#ede9fe',
        'border':'#c4b5fd',
        'desc':'Specialises in brain, spine, and nervous system conditions.',
        'next':[
            'Visit Level 4, Wing A',
            'Estimated wait: 25–35 min',
            'Bring list of current medications'
        ]
    },

    'General Medicine': {
        'icon':'🩺',
        'color':'#059669',
        'bg':'#d1fae5',
        'border':'#6ee7b7',
        'desc':'Handles general health concerns and non-specialist conditions.',
        'next':[
            'Visit Level 1, Wing A',
            'Estimated wait: 10–15 min',
            'Registration desk is open 24/7'
        ]
    },

    'Dermatology': {
        'icon':'🔬',
        'color':'#b45309',
        'bg':'#fef9c3',
        'border':'#fde68a',
        'desc':'Specialises in skin, hair, and nail conditions.',
        'next':[
            'Visit Level 2, Wing D',
            'Estimated wait: 15–20 min',
            'Bring photos of affected area if possible'
        ]
    },
}

# ==========================================================
# HEADER UTAMA APLIKASI
# Ditampilkan di bagian paling atas halaman
# ==========================================================

st.markdown("""
<div style="background:linear-gradient(135deg,#1e3a8a 0%,#1a56db 60%,#0ea5e9 100%);
            padding:3rem 2rem 2.5rem;margin:-1rem -1rem 2rem;text-align:center;">
    <div style="font-size:14px;font-weight:500;color:rgba(255,255,255,0.7);
                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">
        🏥 Future Classroom · Machine Learning
    </div>
    <div style="font-size:36px;font-weight:700;color:#ffffff;margin-bottom:12px;
                letter-spacing:-0.02em;">
        Smart Hospital Patient Navigator
    </div>
    <div style="font-size:18px;color:rgba(255,255,255,0.85);font-weight:400;">
        Find the Right Department for Your Symptoms
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# FORM INPUT PASIEN
# Semua informasi pasien diisi di sini
# ==========================================================

with st.form("triage_form"):

    # ------------------------------------------------------
    # SECTION 1
    # Memilih gejala yang dirasakan pasien
    # ------------------------------------------------------

     st.markdown("""
    <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="background:#0284c7;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">1</span>
            <span style="font-size:16px;font-weight:600;color:#0c4a6e;">What are your main symptoms?</span>
            <span style="font-size:13px;color:#6b7280;font-style:italic;">select all that apply</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Membuat 4 kolom checkbox gejala
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        fever = st.checkbox("🌡️ Fever")
        cough = st.checkbox("🤧 Cough")

    with c2:
        headache = st.checkbox("🤕 Headache")
        chest_pain = st.checkbox("💔 Chest Pain")

    with c3:
        stomach_pain = st.checkbox("🤢 Stomach Pain")
        shortness_breath = st.checkbox("😮‍💨 Shortness of Breath")

    with c4:
        nausea_vomiting = st.checkbox("🤮 Nausea / Vomiting")
        dizziness = st.checkbox("😵 Dizziness")

    c5, _, _, _ = st.columns(4)

    with c5:
        skin_rash = st.checkbox("🔴 Skin Rash")
        st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # SECTION 2
    # Lama gejala dan keluhan utama pasien
    # ------------------------------------------------------

    st.markdown("""
    <div style="background:#fdf4ff;border:1px solid #e9d5ff;
                border-radius:14px;padding:20px 24px;
                margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#7c3aed;color:white;
                         border-radius:8px;padding:4px 10px;
                         font-size:12px;font-weight:600;">2</span>
            <span style="font-size:16px;font-weight:600;color:#3b0764;">
                How long have you had these symptoms?
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cc, col_dur = st.columns(2)

    # Keluhan utama pasien
    with col_cc:
        chief_complaint = st.selectbox(
            "Chief complaint",
            options=list(cc_map.keys())
        )

    # Lama gejala dirasakan
    with col_dur:
        duration = st.selectbox(
            "Duration",
            options=list(dur_map.keys()),
            index=1
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # SECTION 3
    # Tingkat keparahan kondisi pasien
    # ------------------------------------------------------

    st.markdown("""
    <div style="background:#fff7ed;border:1px solid #fed7aa;
                border-radius:14px;padding:20px 24px;
                margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#ea580c;color:white;
                         border-radius:8px;padding:4px 10px;
                         font-size:12px;font-weight:600;">3</span>
            <span style="font-size:16px;font-weight:600;color:#7c2d12;">
                How would you rate the severity?
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_temp, col_hr = st.columns(2)

    # Pilihan suhu tubuh
    with col_temp:
        temperature_level = st.selectbox(
            "Temperature",
            options=list(temp_map.keys()),
            index=1
        )

    # Pilihan kondisi denyut jantung
    with col_hr:
        heart_rate_level = st.selectbox(
            "Heart rate",
            options=list(hr_map.keys()),
            index=1
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # SECTION 4
    # Riwayat penyakit yang dimiliki pasien
    # ------------------------------------------------------

    st.markdown("""
    <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                border-radius:14px;padding:20px 24px;
                margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;
                    margin-bottom:4px;">
            <span style="background:#059669;color:white;
                         border-radius:8px;padding:4px 10px;
                         font-size:12px;font-weight:600;">4</span>
            <span style="font-size:16px;font-weight:600;color:#064e3b;">
                Do you have any of the following?
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ch1, ch2, ch3, _ = st.columns(4)

    with ch1:
        hypertension = st.checkbox("🩺 High Blood Pressure")

    with ch2:
        heart_disease = st.checkbox("❤️ Heart Disease")

    with ch3:
        asthma = st.checkbox("💨 Asthma")

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # SECTION 5
    # Informasi dasar pasien
    # ------------------------------------------------------

    st.markdown("""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;
                border-radius:14px;padding:20px 24px;
                margin-bottom:24px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#475569;color:white;
                         border-radius:8px;padding:4px 10px;
                         font-size:12px;font-weight:600;">5</span>
            <span style="font-size:16px;font-weight:600;color:#1e293b;">
                Patient Information
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_age, col_gen = st.columns(2)

    # Umur pasien
    with col_age:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=35
        )

    # Jenis kelamin pasien
    with col_gen:
        gender = st.selectbox(
            "Gender",
            options=['Female', 'Male']
        )

    # Tombol untuk menjalankan AI
    submitted = st.form_submit_button(
        "Get AI Recommendation →"
    )

# ==========================================================
# PROSES MACHINE LEARNING
# Berjalan setelah tombol ditekan
# ==========================================================

if submitted:

    # Mengubah semua input menjadi format data yang
    # dapat dibaca oleh model Machine Learning
    patient = pd.DataFrame([{
        'age': age,
        'gender': gender_map.get(gender, 0),

        'fever': int(fever),
        'cough': int(cough),
        'headache': int(headache),
        'chest_pain': int(chest_pain),
        'stomach_pain': int(stomach_pain),
        'shortness_breath': int(shortness_breath),
        'nausea_vomiting': int(nausea_vomiting),
        'dizziness': int(dizziness),
        'skin_rash': int(skin_rash),

        'temperature_level':
            temp_map.get(temperature_level, 1),

        'heart_rate_level':
            hr_map.get(heart_rate_level, 1),

        'duration':
            dur_map.get(duration, 1),

        'asthma': int(asthma),
        'hypertension': int(hypertension),
        'heart_disease': int(heart_disease),

        'chief_complaint':
            cc_map.get(chief_complaint, 9)
    }])

    # --------------------------------------------------
    # DATA SCALING
    # Menyamakan skala data dengan data training
    # --------------------------------------------------

    patient_scaled = patient.copy()

    patient_scaled[cols_to_scale] = scaler.transform(
        patient[cols_to_scale]
    )

    # --------------------------------------------------
    # KNN PREDICTION
    # Memprediksi departemen tujuan pasien
    # --------------------------------------------------

    pred = model.predict(
        patient_scaled[features]
    )[0]

    # Menghitung probabilitas untuk setiap departemen
    proba = model.predict_proba(
        patient_scaled[features]
    )[0]

    # Mengambil nama departemen hasil prediksi
    dept_name = dept_map_inv[pred]

    # Confidence score (% keyakinan model)
    confidence = proba[pred] * 100

    # Mengambil informasi departemen
    info = DEPT_INFO[dept_name]

    # ==================================================
    # MENAMPILKAN HASIL AI
    # ==================================================

    st.markdown("---")

    st.markdown("""
    <div style="font-size:22px;
                font-weight:700;
                color:#111827;
                margin-bottom:4px;">
        AI Recommendation
    </div>

    <div style="font-size:14px;
                color:#6b7280;
                margin-bottom:1.5rem;">
        Based on the information you provided
    </div>
    """, unsafe_allow_html=True)

    # Membuat area hasil menjadi 2 kolom
    res_col, prob_col = st.columns([3, 2])

    # ==================================================
    # KOLOM KIRI
    # REKOMENDASI DEPARTEMEN
    # ==================================================

    with res_col:

        # Membuat daftar langkah yang harus dilakukan pasien
        steps_html = ''.join(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="color:{info["color"]};font-size:14px;">📍</span>'
            f'<span style="font-size:14px;color:#374151;">{step}</span></div>'
            for step in info['next']
        )

        # Menampilkan kartu rekomendasi departemen
        st.markdown(f"""
        ...
        """, unsafe_allow_html=True)

    # ==================================================
    # KOLOM KANAN
    # MENAMPILKAN PROBABILITAS SEMUA DEPARTEMEN
    # ==================================================

    with prob_col:

        st.markdown(f"""
        <div style="background:white;
                    border:1px solid #e5e7eb;
                    border-radius:16px;
                    padding:24px;">
            <div style="font-size:14px;
                        font-weight:600;
                        color:#111827;
                        margin-bottom:16px;">
                Confidence by department
            </div>
        """, unsafe_allow_html=True)

        # Mengurutkan departemen berdasarkan
        # probabilitas terbesar
        sorted_depts = sorted(
            dept_map_inv.items(),
            key=lambda x: proba[x[0]],
            reverse=True
        )

        bars_html = ""

        # Membuat progress bar probabilitas
        for idx, dname in sorted_depts:

            pct = proba[idx] * 100

            dinfo = DEPT_INFO[dname]

            is_top = dname == dept_name

            bars_html += f"""
            ...
            """

        st.markdown(
            bars_html +
            """
            <div style="margin-top:20px;
                        background:#eff6ff;
                        border:1px solid #bfdbfe;
                        border-radius:10px;
                        padding:12px 14px;
                        font-size:12px;
                        color:#1e40af;">

                <strong>Model:</strong>
                KNN (k=7)

                · 102,000 patients

                · 99.5% accuracy

                <br>

                <strong>Powered by:</strong>
                Future Classroom
            </div>
        </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================================
# IMPORTANT NOTE
# Hasil AI hanya berupa rekomendasi dan bukan diagnosis.
# Dokter tetap menjadi pihak yang menentukan diagnosis.
# ==========================================================
