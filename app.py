import os
import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Pneumoflag", page_icon="🫁")
st.title("🫁 Pneumoflag Demo")
st.write("흉부 X-ray 이미지를 업로드하면 폐렴 확률과 불확실성(σ), 그리고 Grad-CAM을 보여줍니다.")

default_api = os.environ.get("API_URL", "http://localhost:8000/predict")
API_URL = st.text_input("API URL", default_api)
GRADCAM_URL = API_URL.replace("/predict", "/gradcam")

uploaded = st.file_uploader("X-ray 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("분석하기"):
            with st.spinner("분석 중..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                r = requests.post(API_URL, files=files, timeout=120)

            if r.status_code != 200:
                st.error(f"API 오류: {r.status_code}\n{r.text}")
            else:
                data = r.json()
                st.success("완료!")
                st.metric("Prediction", data["prediction"])
                st.metric("Mean Probability (μ)", data["mean_probability"])
                st.metric("Uncertainty (σ)", data["uncertainty_sigma"])
                st.write("Reject:", "✅ YES" if data["is_rejected"] else "❌ NO")
                st.info(data["message"])

    with col2:
        if st.button("Grad-CAM 보기"):
            with st.spinner("Grad-CAM 생성 중..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                r2 = requests.post(GRADCAM_URL, files=files, timeout=120)

            if r2.status_code != 200:
                st.error(f"Grad-CAM API 오류: {r2.status_code}\n{r2.text}")
            else:
                d2 = r2.json()
                img_bytes = base64.b64decode(d2["heatmap_png_base64"])
                cam_img = Image.open(BytesIO(img_bytes))
                st.success("Grad-CAM 생성 완료!")
                st.metric("Grad-CAM Prediction", d2["prediction"])
                st.metric("Grad-CAM Probability", d2["probability"])
                st.image(
                    cam_img,
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )