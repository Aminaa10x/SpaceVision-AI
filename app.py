import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time

st.set_page_config(page_title="SpaceVision AI", layout="wide")

# ===================== WHITE MODERN STYLE =====================
st.markdown("""
<style>

.stApp {
    background-color: #f4f6f9;
}

html, body, [class*="css"] {
    color: #111111 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

h1 {
    font-size: 38px;
    font-weight: 600;
}

.section-card {
    background: white;
    padding: 36px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
    margin-bottom: 30px;
}

.metric-modern {
    background: white;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}

[data-testid="stFileUploader"] {
    background-color: white !important;
    border-radius: 14px !important;
    border: 1px solid #e5e7eb !important;
}

.footer {
    margin-top: 80px;
    text-align: center;
    color: #6b7280;
}

</style>
""", unsafe_allow_html=True)

# ===================== NAVIGATION =====================
selected = option_menu(
    menu_title=None,
    options=["Home", "Fire AI", "Technology"],
    icons=["house", "flame", "cpu"],
    orientation="horizontal",
)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights="DEFAULT")
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ===================== HOME =====================
if selected == "Home":

    st.markdown("# SpaceVision AI")
    st.markdown("AI негізіндегі орман өртін анықтау жүйесі")

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.write("""
### Жоба туралы

SpaceVision AI – орман өрттерін ерте кезеңде анықтауға арналған 
жасанды интеллект жүйесі. Жоба климаттың өзгеруі салдарынан 
жиілеп бара жатқан өрттер мәселесін шешуге бағытталған.

Өрттер экожүйеге, экономикаға және адам өміріне үлкен қауіп төндіреді. 
Әсіресе шалғай аймақтарда өртті дер кезінде анықтау қиын.

Қазіргі мониторинг жүйелері үлкен көлемдегі спутниктік деректерге сүйенеді, 
алайда оларды қолмен талдау көп уақыт алады.

SpaceVision AI осы процесті автоматтандырады.

---

### Жүйенің негізгі мүмкіндіктері

• Сурет арқылы өртті автоматты анықтау  
• Өрт ықтималдығын пайызбен есептеу  
• Тәуекел деңгейін классификациялау  
• Онлайн веб-интерфейс  
• Жылдам өңдеу және визуализация  

---

### Қолданылатын технология

Жүйе MobileNetV2 нейрондық желісіне негізделген.  
Бұл жеңіл әрі тиімді deep learning архитектурасы.

Модель суретті талдап, Fire / No Fire шешімін шығарады 
және ықтималдық көрсеткішін есептейді.

---

### Неге бұл маңызды?

Ерте анықталған өрт — үлкен апаттың алдын алу мүмкіндігі.

SpaceVision AI:

• Жауап беру уақытын қысқартады  
• Адам факторын азайтады  
• Экологиялық қауіпсіздікті арттырады  
• Ғарыштық деректерді тиімді пайдалануға мүмкіндік береді  

---

### Болашақ даму бағыттары

• Арнайы fire dataset арқылы қайта оқыту  
• YOLO detection интеграциясы  
• Pixel-level сегментация  
• Инфрақызыл деректерді қосу  
• Тікелей спутник ағынымен жұмыс  

SpaceVision AI — бұл табиғи апаттардың алдын алуға 
бағытталған интеллектуалды шешім.
""")
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== FIRE AI =====================
elif selected == "Fire AI":

    st.markdown("# Fire Classification Engine")

    uploaded_file = st.file_uploader("Суретті жүктеңіз", type=["jpg","png","jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Жүктелген сурет")

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            fire_prob = probabilities[0][1].item()

        if fire_prob > 0.8:
            risk = "Қауіпті"
        elif fire_prob > 0.5:
            risk = "Жоғары"
        elif fire_prob > 0.3:
            risk = "Орташа"
        else:
            risk = "Төмен"

        st.markdown("### Өрт ықтималдығы")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fire_prob * 100,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ef4444"},
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.write(f"""
### Нәтиже

Өрт ықтималдығы: {fire_prob*100:.2f}%  
Тәуекел деңгейі: **{risk}**

Жүйе суретті нейрондық желі арқылы талдап,
ықтималдық негізінде шешім шығарады.
""")
        st.markdown("</div>", unsafe_allow_html=True)

# ===================== TECHNOLOGY =====================
elif selected == "Technology":

    st.markdown("# Модель архитектурасы")

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.write("""
SpaceVision AI келесі кезеңдерден тұрады:

1. Суретті алдын ала өңдеу  
2. MobileNetV2 feature extraction  
3. Binary classification  
4. Softmax ықтималдық есептеу  
5. Тәуекел классификациясы  

Бұл архитектура жеңіл, жылдам және 
реал-тайм режиміне бейімделген.
""")
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== FOOTER =====================
st.markdown("""
<div class='footer'>
<hr>
SpaceVision AI © 2026 — AI Fire Detection Platform
</div>
""", unsafe_allow_html=True)