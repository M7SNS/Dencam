import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import base64
import pandas as pd
from datetime import datetime
import hashlib
import io
import time

st.set_page_config(layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "en"

def t(en, ar):
    return ar if st.session_state.lang == "ar" else en

def load_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

icon_b64 = load_base64("icon.png")

if st.button(t("Switch to Arabic", "التبديل إلى الإنجليزية")):
    st.session_state.lang = "ar" if st.session_state.lang == "en" else "en"

st.markdown(f"""
<style>
.header {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
    margin-left: 10px;
}}
.header img {{
    width: 55px;
}}
.header h1 {{
    margin: 0;
    padding: 0;
    font-size: 42px;
}}
</style>

<div class="header">
    <img src="data:image/png;base64,{icon_b64}">
    <h1>Dencam</h1>
</div>
""", unsafe_allow_html=True)

# MODEL
loaded = tf.saved_model.load("model.savedmodel")
infer = loaded.signatures["serving_default"]

class_names = ["Cavity", "Filling", "Impacted", "Implant", "Normal"]
translate = {
    "Cavity": "تسوس",
    "Filling": "حشوة",
    "Impacted": "سن متضرر",
    "Implant": "سن مزروع",
    "Normal": "سن طبيعي"
}
reverse_translate = {v: k for k, v in translate.items()}

if "records" not in st.session_state:
    st.session_state.records = []
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None


def hash_image(img_bytes):
    return hashlib.md5(img_bytes).hexdigest()

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, 0).astype(np.float32)

def classify(img):
    out = infer(tf.constant(preprocess(img)))
    preds = list(out.values())[0].numpy()[0]
    idx = np.argmax(preds)
    eng = class_names[idx]
    conf = float(preds[idx])
    disp = translate[eng] if st.session_state.lang == "ar" else eng
    return disp, conf, eng


# UPLOAD
uploaded = st.file_uploader(t("Upload a Picture", "ارفع صورة"), type=["jpg", "jpeg", "png"])

if uploaded:
    img_bytes = uploaded.getvalue()
    img_hash = hash_image(img_bytes)
    pil_img = Image.open(uploaded).convert("RGB")
    img = np.array(pil_img)

    st.image(pil_img, caption=t("Uploaded Picture", "الصورة المُحملة"), width=300)

    disp, conf, eng = classify(img)
    st.write(t("Tooth State:", "حالة السن:") + f" {disp}")
    st.write(t("Confidence:", "نسبة الدقة:") + f" {conf}")

    if img_hash != st.session_state.last_hash:
        st.session_state.records.append({
            "Tooth State": eng,
            "Confidence": conf,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.session_state.last_hash = img_hash


st.write("---")


# BUILD TABLE
def build_table():
    df = pd.DataFrame(st.session_state.records)

    required = ["Tooth State", "Confidence", "Time"]

    # No data at all → return empty table with correct headers
    if df.empty:
        if st.session_state.lang == "ar":
            return pd.DataFrame(columns=["حالة السن", "نسبة الدقة", "الوقت"])
        else:
            return pd.DataFrame(columns=required)

    # Ensure all required columns exist
    for col in required:
        if col not in df.columns:
            df[col] = ""

    df = df[required]

    # Clean corrupted rows
    df = df[df["Tooth State"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    df = df.fillna("")

    # Translate headers + content
    if st.session_state.lang == "ar":
        df = df.copy()
        df["Tooth State"] = df["Tooth State"].apply(lambda x: translate.get(x, ""))
        df.columns = ["حالة السن", "نسبة الدقة", "الوقت"]
    else:
        df.columns = ["Tooth State", "Confidence", "Time"]

    return df


col_search, col_help = st.columns([3, 2])

with col_search:
    search = st.text_input(t("Search:", "بحث:")).strip().lower()

with col_help:
    if st.session_state.lang == "en":
        st.markdown("""
        ### 🔍 Search Guide
        **Keywords you can use:**
        - Cavity  
        - Filling  
        - Impacted  
        - Implant  
        - Normal  

        **Partial search works**  
        - "cav" → Cavity  
        - "fill" → Filling  
        """)
    else:
        st.markdown("""
        <div style="direction: rtl; text-align: right;">
        <h3>🔍 دليل البحث</h3>

        <b>الكلمات التي يمكنك استخدامها:</b><br>
        - تسوس  
        - حشوة  
        - سن متضرر  
        - سن مزروع  
        - سن طبيعي  

        <b>البحث الجزئي يعمل:</b><br>
        - "تسو" → تسوس  
        - "حشو" → حشوة  
        - "طبي" → سن طبيعي  
        </div>
        """, unsafe_allow_html=True)
def safe_lower(x):
    return x.lower() if isinstance(x, str) else ""

def match_row(row):
    if st.session_state.lang == "ar":
        ar_val = safe_lower(row["حالة السن"])
        eng_val = safe_lower(reverse_translate.get(row["حالة السن"], ""))
    else:
        eng_val = safe_lower(row["Tooth State"])
        ar_val = safe_lower(translate.get(row["Tooth State"], ""))
    return (search in eng_val) or (search in ar_val)


# SEARCH BUTTON
if st.button(t("Search Data", "بحث البيانات")):
    df = build_table()

    if len(df) == 0:
        msg = st.empty()
        msg.error(t("No saved data.", "لا يوجد بيانات."))
        time.sleep(2)
        msg.empty()
    else:
        df = df[df.apply(match_row, axis=1)] if search else df
        if len(df) == 0:
            msg = st.empty()
            msg.error(t("No matching results.", "لا توجد نتائج."))
            time.sleep(2)
            msg.empty()
        else:
            st.dataframe(df)


# SHOW ALL DATA
if st.button(t("Show All Data", "عرض كل البيانات")):
    df = build_table()
    if len(df) == 0:
        msg = st.empty()
        msg.error(t("No saved data.", "لا يوجد بيانات."))
        time.sleep(2)
        msg.empty()
    else:
        st.dataframe(df)


# DELETE BUTTON
if st.button(t("Delete All Data", "حذف جميع البيانات")):
    st.session_state.records = []
    st.session_state.last_hash = None
    msg = st.empty()
    msg.success(t("All data deleted.", "تم حذف جميع البيانات."))
    time.sleep(2)
    msg.empty()


# EXPORT EXCEL
df_excel = build_table()
excel_buffer = io.BytesIO()
df_excel.to_excel(excel_buffer, index=False, engine="openpyxl")
excel_bytes = excel_buffer.getvalue()

st.download_button(
    label=t("Download Excel", "تحميل Excel"),
    data=excel_bytes,
    file_name="teeth_records.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


# ABOUT US
if st.button(t("About Us", "من نحن")):
    if st.session_state.lang == "en":
        st.info("""
We are Dental Guiders, a team passionate about integrating technology with healthcare.
This is our first project, which analyzes X-ray tooth images and classifys each tooth based on its condition.

Team Members:
- Abdulmohsen Al-khaldi
- Ibrahim Al-hamidi
- Makki Zakri
- Azam Al-zeid
""")
    else:
        st.markdown("""
<div style="direction: rtl; text-align: right; font-size: 18px;">
نحن Dental Guiders، فريق مهتم بدمج التقنية مع المجال الصحي.<br>
هذا المشروع يقوم بتحليل صور أشعة الأسنان وتصنيف حالة السن.
<br><br>
<strong>أعضاء الفريق:</strong><br>
- عبدالمحسن الخالدي<br>
- إبراهيم الحميدي<br>
- مكي زكري<br>
- عزام الزيد
</div>
""", unsafe_allow_html=True)
