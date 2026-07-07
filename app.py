import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Dashboard LAB 14 - Morosidad Bancaria",
    layout="wide"
)

st.title("Dashboard Analítico Personalizado - LAB 14")
st.caption("Caso: Riesgo de Morosidad Bancaria | Variable objetivo: moroso")

df = pd.read_csv("dataset_personal.csv")
target = "moroso"

st.subheader("Vista previa del dataset")
st.dataframe(df.head(30), use_container_width=True)

tasa = df[target].mean() * 100
ratio_promedio = df["ratio_deuda_ingreso"].mean()
capacidad_promedio = df["capacidad_pago"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total de registros", f"{len(df):,}")
col2.metric("KPI: tasa de morosidad", f"{tasa:.2f}%")
col3.metric("Ratio deuda/ingreso promedio", f"{ratio_promedio:.2f}")

st.subheader("Visualización 2: Gráfico comparativo")
resumen = df.groupby("tipo_empleo")[target].mean().reset_index()
resumen[target] = resumen[target] * 100
fig_bar = px.bar(
    resumen,
    x="tipo_empleo",
    y=target,
    title="Porcentaje de morosidad por tipo de empleo",
    labels={target: "% morosidad", "tipo_empleo": "Tipo de empleo"}
)
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Visualización 3: Distribución estadística")
variable = st.selectbox(
    "Selecciona una variable numérica",
    ["ratio_deuda_ingreso", "presion_financiera", "capacidad_pago", "ingresos_mensuales", "cuota_mensual", "puntaje_riesgo_financiero"]
)
fig_hist = px.histogram(df, x=variable, title=f"Histograma de {variable}")
st.plotly_chart(fig_hist, use_container_width=True)
fig_box = px.box(df, y=variable, title=f"Boxplot de {variable}")
st.plotly_chart(fig_box, use_container_width=True)

st.subheader("Visualización 4: Heatmap de correlaciones")
corr = df.select_dtypes(include=["int64", "float64"]).corr()
fig_heat = px.imshow(corr, text_auto=True, title="Mapa de correlaciones")
st.plotly_chart(fig_heat, use_container_width=True)

st.subheader("Storytelling de datos")
st.markdown(f"""
### Hallazgos principales
1. La tasa de morosidad estimada en el dataset es de **{tasa:.2f}%**.
2. El tipo de empleo permite observar diferencias en el riesgo de incumplimiento.
3. Variables como `ratio_deuda_ingreso`, `presion_financiera`, `capacidad_pago` y `atrasos_previos` ayudan a explicar el riesgo crediticio.

### Recomendaciones
1. Revisar con mayor cuidado los créditos con alta presión financiera.
2. Ajustar el monto o plazo del crédito cuando la capacidad de pago sea baja.
3. Usar el modelo predictivo como apoyo para clasificar clientes con mayor riesgo de morosidad.
""")
