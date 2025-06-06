import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Carrega o modelo base (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               pooling='avg',
                                               weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Função para gerar embedding facial
def get_embedding(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    embedding = model.predict(img)
    return embedding.flatten()

# Função para calcular similaridade (cosine similarity)
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# Banco de embeddings autorizados
authorized_faces_dir = 'authorized_faces'
authorized_embeddings = {}
for filename in os.listdir(authorized_faces_dir):
    name = os.path.splitext(filename)[0]
    path = os.path.join(authorized_faces_dir, filename)
    authorized_embeddings[name] = get_embedding(path)

# Verificar acesso com base em uma nova imagem
def verify_access(input_image_path, threshold=0.7):
    input_embedding = get_embedding(input_image_path)
    for name, auth_emb in authorized_embeddings.items():
        similarity = cosine_similarity(input_embedding, auth_emb)
        print(f"Comparando com {name} - Similaridade: {similarity:.2f}")
        if similarity > threshold:
            print(f"✅ Acesso Liberado para {name}")
            return True
    print("❌ Acesso Negado")
    return False

# Teste de verificação
verify_access('test_images/entrada.jpg')