import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Configuración de la página
st.set_page_config(
    page_title="Teachable Machine Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Clasificador de Imágenes en Tiempo Real")
st.write("Carga una imagen y el modelo entrenado en Teachable Machine realizará la predicción.")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# FUNCIÓN HÍBRIDA: Resuelve el error de Keras 3 en la nube y Keras 2 en local
@st.cache_resource
def load_teachable_model():
    try:
        # Intenta cargar con parches si el servidor instaló Keras 3
        model = tf.keras.models.load_model("keras_model.h5", compile=False)
        if hasattr(model, 'layers') and len(model.layers) > 0:
            # Forzar la dimensión de entrada correcta si Keras se confunde
            model.layers[0]._batch_input_shape = (None, 224, 224, 3)
        return model
    except Exception:
        try:
            # Alternativa directa estándar
            return tf.keras.models.load_model("keras_model.h5", compile=False)
        except Exception as e:
            st.error(f"Error crítico al cargar el modelo: {e}")
            return None

def load_labels():
    try:
        with open("labels.txt", "r", encoding="utf-8") as f:
            return [line.strip().split(" ", 1)[-1] for line in f.readlines()]
    except FileNotFoundError:
        return []

model = load_teachable_model()
class_names = load_labels()

if model is not None and len(class_names) > 0:
    st.success("✅ ¡Sistema listo y cargado correctamente en internet!")
    
    uploaded_file = st.file_uploader("Selecciona una imagen (JPG, JPEG o PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen cargada", use_container_width=True)
        
        st.write("🔮 *Procesando predicción...*")
        
        # Preprocesamiento estándar
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Inferencia
        prediction = model.predict(data, verbose=0)
        highest_prob_index = np.argmax(prediction[0])
        predicted_class = class_names[highest_prob_index]
        confidence_score = prediction[0][highest_prob_index]
        
        st.markdown("---")
        st.subheader("📊 Resultado del Análisis")
        st.metric(label="Clase Detectada", value=predicted_class, delta=f"{confidence_score * 100:.2f}% Confianza")
        
        # Gráfico
        chart_data = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
        st.bar_chart(chart_data)
else:
    st.warning("⚠️ Asegúrate de tener 'keras_model.h5' y 'labels.txt' en tu repositorio.")