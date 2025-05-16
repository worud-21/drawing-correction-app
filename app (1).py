
import streamlit as st
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

st.set_page_config(page_title="تصحيح الرسم", layout="centered")

st.title("🎨 منصة تصحيح رسومات الشخصيات")

# اختيار الشخصية
character = st.selectbox("🧠 اختر الشخصية التي رسمتها:", [
    "سلحفاة", "غزال", "غراب", "فأر", "صياد"
])

image_map = {
    "سلحفاة": "images/turtle_sample.png",
    "غزال": "images/deer_sample.png",
    "غراب": "images/crow_sample.png",
    "فأر": "images/mouse_sample.png",
    "صياد": "images/hunter_sample.png"
}

uploaded_file = st.file_uploader("📤 ارفع رسمتك هنا (PNG أو JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="📄 رسمتك", use_column_width=True)

    # تحميل صورة الطالب وصورة النموذج
    student_img = Image.open(uploaded_file).convert('L').resize((256, 256))
    reference_img = Image.open(image_map[character]).convert('L').resize((256, 256))

    student_arr = np.array(student_img)
    reference_arr = np.array(reference_img)

    # حساب التشابه
    score, _ = ssim(student_arr, reference_arr, full=True)

    st.write(f"🔎 نسبة التشابه: {score:.2f}")

    if score >= 0.8:
        st.success("🎉 رسم ممتاز! تشابه عالي جدًا، أحسنت!")
    elif score >= 0.4:
        st.info("🙂 جيد! حاول تحسين بعض التفاصيل.")
    else:
        st.error("❌ التشابه منخفض. حاول إعادة الرسم بدقة أكبر.")
