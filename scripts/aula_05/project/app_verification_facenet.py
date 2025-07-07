import streamlit as st
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

EMBEDDING_DIR = 'mean_embeddings'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar modelo FaceNet (pré-treinado no VGGFace2)
@st.cache_resource
def load_models():
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)    
    return mtcnn, facenet

# Função para extrair embedding facial
def get_embedding(mtcnn, facenet, image):
    image = image.convert('RGB')
    face = mtcnn(image)
    if face is None:
        raise ValueError("Nenhum rosto detectado.")
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet(face)
    return embedding.squeeze().cpu().numpy()

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# Função para verificar se o rosto é autorizado
def is_authorized(input_embedding, user_embeddings, threshold=0.7):   
    for user, emb_ref in user_embeddings.items():
        sim = cosine_similarity(input_embedding, emb_ref)
        if sim > threshold:
            return True, user, sim
    return False, None, None

# Load embeddings cadastrados (simulação)   
def load_known_embeddings(embedding_dir):
    embeddings = {}
    for file in os.listdir(embedding_dir):
        if file.endswith(".npy"):
            user = file.replace('.npy', '')
            vector = np.load(os.path.join(embedding_dir, file))
            embeddings[user] = vector
    return embeddings    

# ========== APP Streamlit ==========

st.title("🔐 Verificação Facial de Acesso")

mtcnn, facenet = load_models()
known_embeddings = load_known_embeddings(EMBEDDING_DIR)  

# Upload ou câmera
option = st.radio("Escolha o método de envio da imagem:", ['Upload de Imagem', 'Captura por Webcam'])

if option == 'Upload de Imagem':
    uploaded_file = st.file_uploader("Envie a imagem", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == 'Captura por Webcam':
    camera_image = st.camera_input("Tire uma foto")
    if camera_image:
        image = Image.open(camera_image)

# Verificar acesso
if 'image' in locals():
    st.image(image, caption="Imagem recebida", width=300)

    if st.button("🔍 Verificar acesso"):
        embedding = get_embedding(mtcnn, facenet, image)        
        autorizado, nome, dist = is_authorized(embedding, known_embeddings)

        if autorizado:
            st.success(f"Acesso autorizado! Bem-vindo(a), {nome} (Similaridade = {dist:.3f})")
        else:
            st.error("Acesso negado: rosto não reconhecido.")
