import streamlit as st
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
st.write("Carga una imagen y el modelo entrenado realizará la predicción de forma local.")

# Intentar cargar usando ONNX (Estándar de servidores ligeros)
def load_onnx_model():
    try:
        import onnxruntime as ort
        # Si tienes el modelo convertido a .onnx, lo lee directo
        if os.path.exists("model.onnx"):
            return ort.InferenceSession("model.onnx"), "onnx"
    except Exception:
        pass
    return None, None

# Intentar cargar usando TensorFlow (Si el entorno lo permite en el futuro)
def load_tf_model():
    try:
        import tensorflow as tf
        if os.path.exists("keras_model.h5"):
            model = tf.keras.models.load_model("keras_model.h5", compile=False)
            return model, "tf"
    except Exception:
        pass
    return None, None

# Cargar las etiquetas de clases
def load_labels():
    try:
        with open("labels.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            classes = []
            for line in lines:
                parts = line.strip().split(" ", 1)
                classes.append(parts[1] if len(parts) > 1 else parts[0])
            return classes
    except FileNotFoundError:
        return []

# Inicializar motor de inferencia inteligente
model, motor = load_onnx_model()
if model is None:
    model, motor = load_tf_model()

class_names = load_labels()

# Interfaz de Usuario
if class_names:
    if model is not None:
        st.success(f"✅ ¡Sistema listo en internet! (Motor activo: {motor.upper()})")
    else:
        # Modo de contingencia educativa por si TensorFlow falla en la nube
        st.warning("⚠️ El servidor de la nube no soporta TensorFlow pesado por incompatibilidad de Python.")
        st.info("💡 Para activar el clasificador 100% online gratis: descarga la versión de tu modelo en formato 'TensorFlow Lite' o 'ONNX' desde Teachable Machine, nómbralo 'model.onnx' y súbelo a este GitHub.")
    
    uploaded_file = st.file_uploader("Selecciona una imagen (JPG, JPEG o PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen cargada", use_container_width=True)
        
        st.write("🔮 *Procesando análisis...*")
        
        # Preprocesamiento estándar
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.expand_dims(normalized_image_array, axis=0)
        
        # Ejecutar predicción según el motor disponible
        prediction = None
        if motor == "tf":
            prediction = model.predict(data, verbose=0)[0]
        elif motor == "onnx":
            input_name = model.get_inputs()[0].name
            prediction = model.run(None, {input_name: data})[0][0]
        
        if prediction is not None:
            highest_prob_index = np.argmax(prediction)
            predicted_class = class_names[highest_prob_index]
            confidence_score = prediction[highest_prob_index]
            
            st.markdown("---")
            st.subheader("📊 Resultado del Análisis")
            st.metric(label="Clase Detectada", value=predicted_class, delta=f"{confidence_score * 100:.2f}% Confianza")
            
            chart_data = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
            st.bar_chart(chart_data)
else:
    st.error("❌ No se encontró el archivo 'labels.txt' en tu GitHub.")
