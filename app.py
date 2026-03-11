import os
import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Pneumoflag", page_icon="🫁")

st.title("🫁 Pneumoflag Demo")
st.write("흉부 X-ray 이미지를 업로드하면 폐렴 확률과 불확실성(σ), 그리고 Grad-CAM을 보여줍니다.")

# -----------------------------
# session_state 초기화
# -----------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "gradcam_result" not in st.session_state:
    st.session_state.gradcam_result = None

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

if "analysis_success_message" not in st.session_state:
    st.session_state.analysis_success_message = False

if "gradcam_success_message" not in st.session_state:
    st.session_state.gradcam_success_message = False

# -----------------------------
# API URL 설정
# -----------------------------
default_api_base = os.environ.get("API_BASE_URL", "https://pneumoflag-api.onrender.com")
API_BASE_URL = st.text_input("API Base URL", default_api_base).rstrip("/")

PREDICT_URL = f"{API_BASE_URL}/predict"
GRADCAM_URL = f"{API_BASE_URL}/gradcam"

# -----------------------------
# 파일 업로드
# -----------------------------
uploaded = st.file_uploader("X-ray 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    # 새 파일 업로드 시 이전 상태 초기화
    if st.session_state.last_uploaded_name != uploaded.name:
        st.session_state.analysis_done = False
        st.session_state.analysis_result = None
        st.session_state.gradcam_result = None
        st.session_state.analysis_success_message = False
        st.session_state.gradcam_success_message = False
        st.session_state.last_uploaded_name = uploaded.name

    st.image(uploaded, caption="Uploaded image", use_container_width=True)

    col1, col2 = st.columns(2)

    # -----------------------------
    # 분석하기 버튼
    # -----------------------------
    with col1:
        if st.button("분석하기", use_container_width=True):
            with st.spinner("분석 중..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                try:
                    r = requests.post(PREDICT_URL, files=files, timeout=120)
                except requests.exceptions.RequestException as e:
                    st.error(f"요청 실패: {e}")
                    st.stop()

            if r.status_code != 200:
                st.session_state.analysis_done = False
                st.session_state.analysis_result = None
                st.session_state.analysis_success_message = False
                st.error(f"API 오류: {r.status_code}\n{r.text}")
            else:
                data = r.json()
                st.session_state.analysis_done = True
                st.session_state.analysis_result = data
                st.session_state.gradcam_result = None
                st.session_state.analysis_success_message = True
                st.session_state.gradcam_success_message = False

    # -----------------------------
    # Grad-CAM 보기 버튼
    # -----------------------------
    with col2:
        if st.button("Grad-CAM 보기", use_container_width=True):
            if not st.session_state.analysis_done:
                st.info("먼저 분석하기를 실행하세요.")
            else:
                with st.spinner("Grad-CAM 생성 중..."):
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    try:
                        r2 = requests.post(GRADCAM_URL, files=files, timeout=120)
                    except requests.exceptions.RequestException as e:
                        st.error(f"요청 실패: {e}")
                        st.stop()

                if r2.status_code == 429:
                    retry_after = r2.headers.get("Retry-After")
                    if retry_after:
                        st.error(f"요청이 너무 많습니다. {retry_after}초 후 다시 시도해주세요.")
                    else:
                        st.error("요청이 너무 많습니다. 잠시 후 다시 시도해주세요.")
                elif r2.status_code != 200:
                    st.error(f"Grad-CAM API 오류: {r2.status_code}\n{r2.text}")
                else:
                    d2 = r2.json()
                    st.session_state.gradcam_result = d2
                    st.session_state.gradcam_success_message = True

    # -----------------------------
    # 성공 메시지 표시
    # -----------------------------
    if st.session_state.analysis_success_message:
        st.success("분석 완료!")

    if st.session_state.gradcam_success_message:
        st.success("Grad-CAM 생성 완료!")

    # -----------------------------
    # 분석 결과 표시
    # -----------------------------
    if st.session_state.analysis_result is not None:
        data = st.session_state.analysis_result

        st.metric("Prediction", data["prediction"])
        st.metric("Mean Probability (μ)", data["mean_probability"])
        st.metric("Uncertainty (σ)", data["uncertainty_sigma"])
        st.write("Reject:", "✅ YES" if data["is_rejected"] else "❌ NO")
        st.info(data["message"])

    # -----------------------------
    # Grad-CAM 결과 표시
    # -----------------------------
    if st.session_state.gradcam_result is not None:
        d2 = st.session_state.gradcam_result

        st.metric("Grad-CAM Prediction", d2["prediction"])
        st.metric("Grad-CAM Probability", d2["probability"])

        img_bytes = base64.b64decode(d2["heatmap_png_base64"])
        cam_img = Image.open(BytesIO(img_bytes))
        st.image(
            cam_img,
            caption="Grad-CAM Overlay",
            use_container_width=True
        )

else:
    st.info("이미지를 먼저 업로드하세요.")