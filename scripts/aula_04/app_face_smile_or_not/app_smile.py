import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Configuração
st.set_page_config(page_title="Detector de Sorriso", layout="centered")

st.title("😄 Detector de Sorriso com Deep Learning")
st.write("Envie uma imagem ou capture pela webcam para saber se a pessoa está sorrindo ou não.")

# Tamanho padrão da imagem para o modelo
IMAGE_SIZE = (150, 150)

# Carrega o modelo com cache
@st.cache_resource
def load_trained_model():
    return load_model("best_model.h5")

model = load_trained_model()

# Função para pré-processar a imagem (OpenCV)
def prepare_image_cv(img_rgb):
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Interface de seleção
option = st.radio("Escolha o método de entrada:", ["📁 Enviar imagem", "📷 Usar webcam"])

if option == "📁 Enviar imagem":
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption="Imagem enviada", use_column_width=True)

            img_array = prepare_image_cv(img_rgb)
            prediction = model.predict(img_array)[0][0]
            label = "😊 Sorrindo" if prediction >= 0.5 else "😐 Não sorrindo"
            confidence = prediction if prediction >= 0.5 else 1 - prediction

            st.markdown(f"### Resultado: **{label}**")
            st.markdown(f"Confiança: **{confidence:.2%}**")
        except Exception as e:
            st.error(f"Erro ao processar a imagem: {e}")

elif option == "📷 Usar webcam":
    st.info("A imagem será capturada pela câmera do seu dispositivo.")
    picture = st.camera_input("Capture uma imagem")

    if picture is not None:
        try:
            file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption="Imagem capturada", use_column_width=True)

            img_array = prepare_image_cv(img_rgb)
            prediction = model.predict(img_array)[0][0]
            label = "😊 Sorrindo" if prediction >= 0.5 else "😐 Não sorrindo"
            confidence = prediction if prediction >= 0.5 else 1 - prediction

            st.markdown(f"### Resultado: **{label}**")
            st.markdown(f"Confiança: **{confidence:.2%}**")
        except Exception as e:
            st.error(f"Erro ao processar a imagem da câmera: {e}")