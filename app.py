import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# reconstruir modelo igual ao do treino
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)

# carregar pesos
state = torch.load("modelo_gato_cachorro.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

classes = ["gato", "cachorro"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("Classificador de Gato ou Cachorro 🐱🐶")

imagem = st.file_uploader("Envie uma imagem:", type=["jpg", "jpeg", "png"])

if imagem:
    img = Image.open(imagem).convert("RGB")
    st.image(img, width=300)

    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img_t)
        _, pred = torch.max(out, 1)

    st.success(f"Resultado: **{classes[pred.item()]}**")
