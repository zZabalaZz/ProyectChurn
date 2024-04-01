import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb

# Cargar modelos
@st.cache_resource()
def load_models():
    modeloNB = jb.load('modeloNB.bin')
    modeloArbol = jb.load('ModeloArbol.bin')
    modeloBosque = jb.load('ModeloBosque.bin')
    return modeloNB, modeloArbol, modeloBosque

modeloNB, modeloArbol, modeloBosque = load_models()

# Definir la función para seleccionar
def seleccionar():
    st.title("Aplicación de predicción")
    st.header('Machine Learning para Churn')
    st.subheader('Ejemplo en los modelos: Naive Bayes, Árbol de Decisión, y Bosque Aleatorio')

    st.subheader("Seleccionar datos de entrada")
    modelo_seleccionado = st.selectbox("Modelo", ['Naive Bayes', 'Árbol de Decisión', 'Bosque Aleatorio'])

    COMP = st.slider("COMP", 4000, 18000, 8000, 100)
    PROM = st.slider("PROM", 0.7, 9.0, 5.0, 0.5)
    COMINT = st.slider("COMINT", 1500, 58000, 12000, 100)
    COMPPRES = st.slider('COMPPRES', 17000, 90000, 25000, 100)
    RATE = st.slider("RATE", 0.5, 4.2, 2.0, 0.1)
    DIASSINQ = st.slider("DIASSINQ", 270, 1800, 500, 10)
    TASARET = st.slider("TASARET", 0.3, 1.9, 0.8, 0.5)
    NUMQ = st.slider("NUMQ", 3.0, 10.0, 4.0, 0.5)
    RETRE = st.number_input("RETRE entre 3 y 30", value=3.3, min_value=3.0, max_value=30.0, step=0.5)

    st.write("*NOTA: Después de seleccionar los datos, desplácese hacia abajo para ver los resultados de predicción.*")

    st.subheader("Predicción")
    st.write("El siguiente es el pronóstico de la deserción usando el modelo", modelo_seleccionado)
    st.write("Se han seleccionado los siguientes parámetros:")
    datos_entrada = {'COMP': [COMP], 'PROM': [PROM], 'COMINT': [COMINT], 'COMPPRES': [COMPPRES], 'RATE': [RATE], 'DIASSINQ': [DIASSINQ], 'TASARET': [TASARET], 'NUMQ': [NUMQ], 'RETRE': [RETRE]}
    df_datos_entrada = pd.DataFrame(datos_entrada)
    st.dataframe(df_datos_entrada)

    # Realizar la predicción
    if modelo_seleccionado == 'Naive Bayes':
        y_predict = modeloNB.predict(df_datos_entrada)
        probabilidad = modeloNB.predict_proba(df_datos_entrada)
    elif modelo_seleccionado == 'Árbol de Decisión':
        y_predict = modeloArbol.predict(df_datos_entrada)
        probabilidad = modeloArbol.predict_proba(df_datos_entrada)
    else:
        y_predict = modeloBosque.predict(df_datos_entrada)
        probabilidad = modeloBosque.predict_proba(df_datos_entrada)

    # Mostrar resultados de predicción
    st.write("Predicción:")
    if y_predict[0] == 1:
        st.write("Resultado: Cliente se retirará")
    else:
        st.write("Resultado: Cliente no se retirará")

    st.write("Con la siguiente probabilidad:")
    col1, col2= st.columns(2)
    col1.metric(label="Probalidad de NO :", value="{0:.2%}".format(probabilidad[0][0]),delta=" ")
    col2.metric(label="Probalidad de SI:", value="{0:.2%}".format(probabilidad[0][1]),delta=" ")
    st.write("La importancia de cada Factor en el modelo es:")
    if modelo_seleccionado != 'Naive Bayes':
        importancia = [0.1, 0.3, 0.2, 0.15, 0.05, 0.08, 0.07, 0.1, 0.1]  # Esto es un ejemplo, debes calcular la importancia real de tus features
        importancia_df = pd.DataFrame({'Importancia': importancia}, index=df_datos_entrada.columns)
        st.bar_chart(importancia_df)
    else:
        st.write("Naive Bayes no tiene parámetro de importancia de los features")


def creditos():
    # Borrar el contenido anterior

    # Nuevo título
    st.title("Créditos")

    st.subheader("**Hecho por:**")
    st.write("**Cristian David Zabala Tavera**")
    # Imagen de introducción
    st.image("introduccion.png")
    st.write("Sara Lucía Uribe Ruiz")
    st.subheader("**Presentado a:**")
    st.write("Alfredo Diaz Claro, Inteligencia Artificial, 55906")
# Función para mostrar el encabezado y el contenido inicial
def mostrar_inicio():
    # Título inicial
    st.title("Análisis Predictivo de Churn")

    # Header inicial
    st.header("Proyecto de Primer Corte")

    # Subheader inicial
    st.write("""
    En el entorno altamente competitivo de las empresas modernas, la retención de clientes es crucial para el éxito a largo plazo. El Churn, o la pérdida de clientes, representa uno de los mayores desafíos para las organizaciones, ya que puede tener un impacto significativo en los ingresos y la rentabilidad. Identificar y prevenir la deserción de clientes es fundamental para mantener una base de clientes sólida y sostenible.

    En este proyecto, nos proponemos desarrollar un sistema de predicción de Churn utilizando técnicas de inteligencia artificial. Nos centraremos en algoritmos de aprendizaje supervisado como Naïve Bayes, árboles de decisión y bosques aleatorios para construir nuestro modelo predictivo. Utilizaremos datos históricos de clientes para entrenar y validar estos modelos, asegurando su precisión y eficacia.

    Objetivos del Proyecto

    - Implementar modelos de IA como Naïve Bayes, árboles de decisión y bosques aleatorios para predecir el Churn de clientes.
    - Utilizar datos históricos de clientes para entrenar y validar los modelos, optimizando su rendimiento.
    - Desarrollar una interfaz de usuario intuitiva que permita a los usuarios ingresar datos de clientes y obtener predicciones de Churn en tiempo real.
    - Evaluar el rendimiento de los modelos en términos de precisión, sensibilidad y especificidad, y realizar ajustes según sea necesario.
    - Proporcionar recomendaciones basadas en los resultados de la predicción para ayudar a las empresas a retener a sus clientes de manera efectiva.

    Beneficios del Proyecto

    - Facilitar el aprendizaje y la comprensión de técnicas de IA como Naïve Bayes, árboles de decisión y bosques aleatorios en un contexto práctico.
    - Proporcionar a las empresas herramientas poderosas para anticipar y prevenir la deserción de clientes, mejorando así su rentabilidad y su relación con los clientes.
    - Fomentar la adopción de prácticas de análisis de datos avanzadas y la inteligencia artificial en el ámbito empresarial.
    """)

# Verificar si es la primera ejecución de la app
if 'primer_ejecucion' not in st.session_state:
    st.session_state.primer_ejecucion = True
    mostrar_inicio()  # Mostrar el contenido inicial si es la primera ejecución

# Mostrar la sección seleccionada
if st.session_state.primer_ejecucion:
    st.session_state.primer_ejecucion = False

# Define las opciones de navegación
options = ["Inicio", "Predecir", "Créditos"]
selection = st.sidebar.selectbox("Navegar", options)

# Mostrar la sección seleccionada
if selection == "Inicio":
    mostrar_inicio()
elif selection == "Predecir":
    seleccionar()
elif selection == "Créditos":
    creditos()