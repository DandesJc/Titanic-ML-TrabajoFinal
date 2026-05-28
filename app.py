import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Titanic ML Dashboard",
    page_icon="🚢",
    layout="wide"
)

# =========================================================
# CARGA DE MODELO Y DATOS
# =========================================================

@st.cache_data
def cargar_dataset():
    return sns.load_dataset("titanic")

@st.cache_resource
def cargar_modelos():
    modelo = joblib.load("mejor_modelo_titanic.pkl")
    scaler = joblib.load("scaler_titanic.pkl")
    return modelo, scaler

df = cargar_dataset()
modelo, scaler = cargar_modelos()

# =========================================================
# TÍTULO PRINCIPAL
# =========================================================

st.title("🚢 Dashboard Machine Learning - Titanic")
st.markdown("""
### Predicción de Supervivencia en el Titanic
Proyecto desarrollado bajo metodología **CRISP-DM**
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navegación")

seccion = st.sidebar.radio(
    "Ir a:",
    [
        "📊 Resumen General",
        "📈 Visualizaciones",
        "🤖 Predicción",
        "📋 Conclusiones"
    ]
)

# =========================================================
# RESUMEN GENERAL
# =========================================================

if seccion == "📊 Resumen General":

    st.header("📊 Resumen General del Dataset")

    total = len(df)
    sobrevivieron = df["survived"].sum()
    no_sobrevivieron = total - sobrevivieron
    tasa = (sobrevivieron / total) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Pasajeros", total)
    col2.metric("Sobrevivieron", sobrevivieron)
    col3.metric("No sobrevivieron", no_sobrevivieron)
    col4.metric("Tasa de supervivencia", f"{tasa:.1f}%")

    st.markdown("---")

    st.subheader("📄 Vista previa del dataset")
    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("---")

    st.subheader("📌 Valores faltantes")

    faltantes = df.isnull().sum()
    faltantes = faltantes[faltantes > 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    faltantes.plot(kind="bar", ax=ax)

    ax.set_ylabel("Cantidad")
    ax.set_title("Valores faltantes por variable")

    st.pyplot(fig)

# =========================================================
# VISUALIZACIONES
# =========================================================

elif seccion == "📈 Visualizaciones":

    st.header("📈 Análisis Visual")

    # =====================================================
    # SUPERVIVENCIA POR GÉNERO
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("👨👩 Supervivencia por género")

        surv_gender = df.groupby("sex")["survived"].mean()

        fig, ax = plt.subplots(figsize=(5,4))
        surv_gender.plot(kind="bar", ax=ax)

        ax.set_ylabel("Tasa de supervivencia")
        ax.set_ylim(0,1)

        st.pyplot(fig)

    # =====================================================
    # SUPERVIVENCIA POR CLASE
    # =====================================================

    with col2:

        st.subheader("🎟️ Supervivencia por clase")

        surv_class = df.groupby("pclass")["survived"].mean()

        fig, ax = plt.subplots(figsize=(5,4))
        surv_class.plot(kind="bar", ax=ax)

        ax.set_ylabel("Tasa de supervivencia")
        ax.set_ylim(0,1)

        st.pyplot(fig)

    st.markdown("---")

    # =====================================================
    # EDAD
    # =====================================================

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("📊 Distribución de edades")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.histplot(
            data=df,
            x="age",
            bins=30,
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

    # =====================================================
    # FARE
    # =====================================================

    with col4:

        st.subheader("💰 Distribución de tarifas")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.histplot(
            data=df,
            x="fare",
            bins=40,
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

    st.markdown("---")

    # =====================================================
    # MATRIZ DE CORRELACIÓN
    # =====================================================

    st.subheader("🔥 Matriz de correlación")

    df_corr = df.copy()

    df_corr["sex"] = df_corr["sex"].map({
        "male": 1,
        "female": 0
    })

    corr = df_corr.select_dtypes(include=np.number).corr()

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# =========================================================
# PREDICCIÓN
# =========================================================

elif seccion == "🤖 Predicción":

    st.header("🤖 Predicción de Supervivencia")

    st.markdown("""
    Ingrese los datos del pasajero para estimar la probabilidad de supervivencia.
    """)

    col1, col2 = st.columns(2)

    with col1:

        pclass = st.selectbox(
            "Clase del pasajero",
            [1, 2, 3]
        )

        sex = st.selectbox(
            "Sexo",
            ["female", "male"]
        )

        age = st.slider(
            "Edad",
            0,
            80,
            30
        )

        fare = st.number_input(
            "Tarifa pagada",
            0.0,
            600.0,
            50.0
        )

    with col2:

        sibsp = st.number_input(
            "Hermanos / Cónyuge",
            0,
            10,
            0
        )

        parch = st.number_input(
            "Padres / Hijos",
            0,
            10,
            0
        )

        embarked = st.selectbox(
            "Puerto de embarque",
            ["C", "Q", "S"]
        )

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    family_size = sibsp + parch
    is_alone = 1 if family_size == 0 else 0

    sex_male = 1 if sex == "male" else 0
    embarked_Q = 1 if embarked == "Q" else 0
    embarked_S = 1 if embarked == "S" else 0

    # =====================================================
    # DATAFRAME FINAL
    # =====================================================

    nuevo = pd.DataFrame([{
        "pclass": pclass,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "family_size": family_size,
        "is_alone": is_alone,
        "sex_male": sex_male,
        "embarked_Q": embarked_Q,
        "embarked_S": embarked_S
    }])

    cols_escalar = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "family_size"
    ]

    nuevo[cols_escalar] = scaler.transform(
        nuevo[cols_escalar]
    )

    # =====================================================
    # PREDICCIÓN
    # =====================================================

    if st.button("🔮 Realizar predicción"):

        pred = modelo.predict(nuevo)[0]
        prob = modelo.predict_proba(nuevo)[0]

        st.markdown("---")

        if pred == 1:
            st.success(
                f"✅ El pasajero probablemente SOBREVIVIRÍA"
            )
        else:
            st.error(
                f"❌ El pasajero probablemente NO SOBREVIVIRÍA"
            )

        st.subheader("📊 Probabilidades")

        colA, colB = st.columns(2)

        colA.metric(
            "Probabilidad de supervivencia",
            f"{prob[1]*100:.2f}%"
        )

        colB.metric(
            "Probabilidad de fallecimiento",
            f"{prob[0]*100:.2f}%"
        )

        st.markdown("---")

        # =================================================
        # INTERPRETACIÓN
        # =================================================

        st.subheader("🧠 Interpretación")

        insights = []

        if sex == "female":
            insights.append(
                "Las mujeres tuvieron tasas históricas más altas de supervivencia."
            )

        if pclass == 1:
            insights.append(
                "Los pasajeros de primera clase tuvieron más acceso a botes salvavidas."
            )

        if age < 12:
            insights.append(
                "Los niños tenían prioridad durante la evacuación."
            )

        if fare > 100:
            insights.append(
                "Tarifas altas suelen asociarse con clases superiores."
            )

        if is_alone == 1:
            insights.append(
                "Viajar solo tuvo comportamientos distintos frente a familias grandes."
            )

        for i in insights:
            st.info(i)

# =========================================================
# CONCLUSIONES
# =========================================================

elif seccion == "📋 Conclusiones":

    st.header("📋 Conclusiones del Proyecto")

    st.markdown("""
    ## 🔍 Hallazgos principales

    - Las mujeres tuvieron mayor tasa de supervivencia.
    - Los pasajeros de primera clase tuvieron ventajas importantes.
    - La edad y la tarifa pagada influyeron significativamente.
    - El dataset presenta sesgos históricos y sociales.

    ---

    ## 🤖 Modelos utilizados

    - Regresión Logística
    - Árbol de Decisión
    - KNN

    ---

    ## ⚠️ Consideraciones éticas

    Este modelo reproduce patrones históricos reales del Titanic,
    incluyendo desigualdades sociales y de género.

    No debe utilizarse para tomar decisiones reales sobre personas.

    ---

    ## 🚀 Deployment

    El modelo fue serializado usando:
    - joblib
    - Streamlit
    - Scikit-learn
    """)

    st.success("✅ Proyecto CRISP-DM completado correctamente")