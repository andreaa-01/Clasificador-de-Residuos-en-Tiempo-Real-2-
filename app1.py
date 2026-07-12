import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Configuración profesional de la página de Streamlit
st.set_page_config(
    page_title="Teachable Machine Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Clasificador de Imágenes en Tiempo Real")
st.write("Carga una imagen y el modelo entrenado en Teachable Machine realizará la predicción.")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Función optimizada con caché para cargar el modelo una sola vez
@st.cache_resource
def load_teachable_model():
    try:
        model = tf.keras.models.load_model("keras_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error crítico al cargar 'keras_model.h5': {e}")
        return None

# Cargar las etiquetas de clases
def load_labels():
    try:
        with open("labels.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            classes = []
            for line in lines:
                parts = line.strip().split(" ", 1)
                if len(parts) > 1:
                    classes.append(parts[1])
                else:
                    classes.append(parts[0])
            return classes
    except FileNotFoundError:
        st.error("No se encontró el archivo 'labels.txt' en la ruta actual.")
        return []

# Inicializar modelo y etiquetas
model = load_teachable_model()
class_names = load_labels()

if model is not None and len(class_names) > 0:
    st.success("✅ ¡Sistema listo y cargado correctamente en internet!")
    
    uploaded_file = st.file_uploader("Selecciona una imagen (JPG, JPEG o PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen cargada para análisis", use_container_width=True)
        
        st.write("🔮 *Procesando predicción...*")
        
        # --- PREPROCESAMIENTO ESTÁNDAR DE TEACHABLE MACHINE ---
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # --- EJECUTAR INFERENCIA ---
        prediction = model.predict(data, verbose=0)
        highest_prob_index = np.argmax(prediction[0])
        predicted_class = class_names[highest_prob_index]
        confidence_score = prediction[0][highest_prob_index]
        
        # --- MOSTRAR RESULTADOS ---
        st.markdown("---")
        st.subheader("📊 Resultado del Análisis")
        st.metric(label="Clase Detectada", value=predicted_class, delta=f"{confidence_score * 100:.2f}% Confianza")
        
        st.write("### Desglose de Probabilidades:")
        chart_data = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
        st.bar_chart(chart_data)
else:
    st.warning("⚠️ Asegúrate de tener 'keras_model.h5' y 'labels.txt' en tu repositorio.")
