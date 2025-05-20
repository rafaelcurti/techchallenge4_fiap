# app.py – versão revisada com correções finais

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from datetime import timedelta
import os

# ====================== CONFIGURAÇÃO DE PÁGINA ======================
st.set_page_config(page_title="Dashboard Petróleo Brent", layout="wide")
st.markdown(
    """
    <style>
        .main { background-color: #f9f9f9; }
        .stButton>button { background-color: #0E1117; color: white; font-weight: bold; }
        .stTabs [role="tab"]{ font-size:1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================== FUNÇÕES AUXILIARES ==========================
@st.cache_data
def carregar_dados():
    df = pd.read_excel("data/IPEA_DB.xlsx")
    df.columns = ["data", "preco"]
    df = df.sort_values("data")
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def carregar_eventos():
    df_eventos = pd.read_csv("data/eventos_petroleo.csv", encoding="latin1")
    df_eventos["data"] = pd.to_datetime(df_eventos["data"])
    return df_eventos

@st.cache_data
def carregar_modelo(nome):
    caminho = f"modelos/{nome}"
    if os.path.exists(caminho):
        with open(caminho, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def carregar_scaler():
    with open("modelos/scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def carregar_lstm():
    return load_model("modelos/melhor_modelo_lstm.keras")

# ====================== DADOS BASE COMPLETA =========================
df = carregar_dados()           # base completa (necessária para eventos)
df_eventos = carregar_eventos() # eventos + descrição

dias = (df["data"] - df["data"].min()).dt.days.values.reshape(-1, 1)

# ====================== MODELOS PARA COMPARAÇÃO =====================
modelos = {
    "Linear Regression": LinearRegression().fit(dias, df["preco"]),
    "Decision Tree":     DecisionTreeRegressor().fit(dias, df["preco"]),
    "Random Forest":      carregar_modelo("melhor_modelo.pkl"),
}

# ====================== INTERFACE – TABS ============================
tabs = st.tabs(["Previsão", "Modelos", "Eventos Históricos", "Desenvolvedores", "Documentação"])

# -------------------------------------------------------------------
# 🔮 PREVISÃO
# -------------------------------------------------------------------
with tabs[0]:
    st.title("🔮 Previsão do Preço do Petróleo Brent")
    modelo_opcao = st.selectbox(
        "Escolha o modelo para previsão:",
        ["Modelo Tradicional (Random Forest)", "Modelo LSTM"],
    )
    dias_previsao = st.slider("Dias a prever:", 1, 10, 5)

    if st.button("Prever"):
        try:
            if modelo_opcao.startswith("Modelo Tradicional"):
                modelo = carregar_modelo("melhor_modelo.pkl")
                ultimo_dia = dias[-1][0]
                dias_futuros = np.array([[ultimo_dia + i] for i in range(1, dias_previsao + 1)])
                previsoes = modelo.predict(dias_futuros)
            else:  # LSTM
                lstm = carregar_lstm()
                scaler = carregar_scaler()
                window = 10
                serie = df["preco"].values.reshape(-1, 1)
                serie_scaled = scaler.transform(serie)
                ultimos = serie_scaled[-window:]
                previsoes = []
                for _ in range(dias_previsao):
                    entrada = np.expand_dims(ultimos, axis=0)
                    pred = lstm.predict(entrada, verbose=0)
                    previsoes.append(pred[0, 0])
                    ultimos = np.append(ultimos[1:], [[pred[0, 0]]], axis=0)
                previsoes = scaler.inverse_transform(np.array(previsoes).reshape(-1, 1)).flatten()

            datas_futuras = [df["data"].max() + timedelta(days=i) for i in range(1, dias_previsao + 1)]
            df_previsao = pd.DataFrame(
                {
                    "Data": [d.strftime("%d/%m/%y") for d in datas_futuras],
                    "Valor Previsto": np.round(previsoes, 2),
                }
            )
            st.success(
                f"Previsão realizada a partir de {df['data'].max().strftime('%d/%m/%Y')}."
            )
            st.dataframe(df_previsao, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")

# -------------------------------------------------------------------
# 📊 MODELOS – avaliação & comparação
# -------------------------------------------------------------------
with tabs[1]:
    st.title("📊 Avaliação de Modelos")

    # Cálculo de métricas incluindo LSTM
    st.subheader("Métricas")
    metricas_modelos = []

    for nome, modelo in modelos.items():
        if modelo is not None:
            y_pred = modelo.predict(dias)
            metricas_modelos.append([
                nome,
                round(mean_squared_error(df["preco"], y_pred), 2),
                round(mean_absolute_error(df["preco"], y_pred), 2),
                round(r2_score(df["preco"], y_pred), 2),
            ])

    # LSTM
    try:
        lstm = carregar_lstm()
        scaler = carregar_scaler()
        window = 10
        serie = df["preco"].values.reshape(-1, 1)
        serie_scaled = scaler.transform(serie)
        X_lstm = np.array([serie_scaled[i - window : i] for i in range(window, len(serie_scaled))])
        y_lstm_true = serie[window:].flatten()
        y_lstm_pred = scaler.inverse_transform(lstm.predict(X_lstm, verbose=0)).flatten()
        metricas_modelos.append([
            "LSTM",
            round(mean_squared_error(y_lstm_true, y_lstm_pred), 2),
            round(mean_absolute_error(y_lstm_true, y_lstm_pred), 2),
            round(r2_score(y_lstm_true, y_lstm_pred), 2),
        ])
    except Exception:
        pass

    df_metricas = pd.DataFrame(metricas_modelos, columns=["Modelo", "MSE", "MAE", "R²"])
    st.dataframe(df_metricas, use_container_width=True, hide_index=True)

    # Comparação linha‑a‑linha
    st.subheader("Comparar valores reais vs previstos (histórico)")
    comparacao = pd.DataFrame(
        {"Data": df["data"].dt.strftime("%d/%m/%y"), "Real": df["preco"].round(2)}
    )
    for nome, modelo in modelos.items():
        if modelo is not None:
            comparacao[nome] = np.round(modelo.predict(dias), 2)

    try:
        comparacao["LSTM"] = [np.nan] * window + list(np.round(y_lstm_pred, 2))
    except:
        pass

    st.dataframe(comparacao.tail(10), use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# 📅 EVENTOS HISTÓRICOS
# -------------------------------------------------------------------
with tabs[2]:
    st.title("📅 Eventos Históricos vs Preço")
    df_eventos["evento_dropdown"] = (
        df_eventos["data"].dt.strftime("%d/%m/%Y") + " – " + df_eventos["evento"]
    )
    evento_label = st.selectbox("Selecione um evento:", df_eventos["evento_dropdown"].tolist())
    evento_row = df_eventos[df_eventos["evento_dropdown"] == evento_label].iloc[0]
    evento_data = evento_row["data"]
    descricao = evento_row.get("descricao", "")
    janela = st.slider("Dias antes e depois do evento", 5, 60, 15)

    df_plot = df[
        (df["data"] >= evento_data - pd.Timedelta(days=janela))
        & (df["data"] <= evento_data + pd.Timedelta(days=janela))
    ].copy()

    if df_plot.empty:
        st.warning("Não há dados históricos suficientes nesse intervalo para exibir.")
    else:
        df_plot["Data"] = df_plot["data"].dt.strftime("%d/%m/%Y")
        df_plot["Preço"] = df_plot["preco"].round(2)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_plot["data"], df_plot["Preço"], marker="o")
        ax.axvline(evento_data, color="red", linestyle="--", label="Evento")
        ax.set_title(f"Preço do Petróleo • Evento: {evento_row['evento']}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Preço (US$)")
        ax.legend()
        st.pyplot(fig)

        if descricao:
            st.markdown(f"**Resumo do Evento:** {descricao}")

        st.dataframe(
            df_plot[["Data", "Preço"]], use_container_width=True, hide_index=True
        )

# -------------------------------------------------------------------
# 👨‍💻 DESENVOLVEDORES
# -------------------------------------------------------------------
with tabs[3]:
    st.title("👨‍💻 Desenvolvedores")
    st.markdown(
        """
        - Ozir José Azevedo Junior 
        - Paloma Cristina Pinheiro
        - Rafael Curti Barros
        - Rilciane de Sousa Bezerra
        """
    )

# -------------------------------------------------------------------
# 📚 DOCUMENTAÇÃO
# -------------------------------------------------------------------
with tabs[4]:
    st.title("📚 Documentação do Projeto")
    st.subheader("1. Análise de Cenário")
    st.image("data/print_analise.png")

    st.subheader("2. Metodologia Aplicada")
    st.image("data/print_metodologia.png")

    st.subheader("3. Treinamento dos Modelos")
    st.image("data/print_modelos.png")
