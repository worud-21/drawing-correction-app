import streamlit as st
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

st.title("🖌️ أداة تصحيح رسم الطالب تلقائيًا")

st.write("### خطوات الاستخدام:")
st.markdown("""
1. 📤 ارفع الصورة **المرجعية** (الرسم الصحيح).
2. 🧑‍🎓 ارفع **رسم الطالب**.
3. 🔍 ستظهر لك نسبة التشابه والفرق بين الرسمتين، مع تغذية راجعة.
""")

ref_file = st.file_uploader("📌 ارفع الصورة المرجعية", type=["png", "jpg", "jpeg"])
student_file = st.file_uploader("✏️ ارفع رسم الطالب", type=["png", "jpg", "jpeg"])

if ref_file and student_file:
    # فتح الصور وتحويلها إلى رمادية
    ref_img = Image.open(ref_file).convert("L")
    student_img = Image.open(student_file).convert("L")

    # تغيير حجم رسم الطالب ليطابق المرجعية
    student_resized = student_img.resize(ref_img.size)

    # تحويل الصور إلى مصفوفات
    ref_np = np.array(ref_img)
    student_np = np.array(student_resized)

    # حساب SSIM
    score, diff = ssim(ref_np, student_np, full=True)
    diff = (diff * 255).astype("uint8")

    # عرض الدرجة
    st.write(f"🔎 **درجة التشابه:** {score * 100:.2f}%")

    # تغذية راجعة مخصصة
    if score >= 0.8:
        st.success("🎉 رسم ممتاز! تشابه عالي جدًا، أحسنت!")
    elif score >= 0.4:
        st.info("🙂 جيد! حاول تحسين بعض التفاصيل مثل الأطراف أو الشكل العام.")
    else:
        st.error("❌ التشابه منخفض. حاول إعادة الرسم بدقة أكبر وركز على الملامح الأساسية.")

    # عرض الصور والفرق
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(ref_np, cmap='gray')
    ax[0].set_title("الصورة المرجعية")
    ax[0].axis("off")

    ax[1].imshow(student_np, cmap='gray')
    ax[1].set_title("رسم الطالب")
    ax[1].axis("off")

    ax[2].imshow(diff, cmap='gray')
    ax[2].set_title("الفرق")
    ax[2].axis("off")

    st.pyplot(fig)
