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
    try:
        return _ler_eventos("utf-8")
    except UnicodeDecodeError:
        return _ler_eventos("latin1")

def _ler_eventos(enc):
    df_eventos = pd.read_csv("data/eventos_petroleo.csv", encoding=enc)
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
df = carregar_dados()
df_eventos = carregar_eventos()

dias = (df["data"] - df["data"].min()).dt.days.values.reshape(-1, 1)

# ====================== MODELOS PARA COMPARAÇÃO =====================
modelos = {
    "Linear Regression": LinearRegression().fit(dias, df["preco"]),
    "Decision Tree":     DecisionTreeRegressor().fit(dias, df["preco"]),
    "Random Forest":     carregar_modelo("melhor_modelo.pkl"),
    "LSTM":              "LSTM",  # LSTM tratado separadamente
}

# ====================== INTERFACE – TABS ============================
tabs = st.tabs([
    "Roteiro de Desenvolvimento",
    "Modelos",
    "Previsão",
    "Eventos Históricos",
    "Time Responsável",
])

# -------------------------------------------------------------------
# Desenvolvimento do Projeto
# -------------------------------------------------------------------
with tabs[0]:
    st.title("📚 Roteiro de Desenvolvimento")

    st.markdown("""
    ## Introdução
    O presente projeto visa desenvolver uma aplicação analítica utilizando dados históricos do preço do petróleo tipo Brent, correlacionando com eventos históricos marcantes que afetaram direta ou indiretamente a cotação da commodity.

    ## Objetivo
    Desenvolver um modelo de previsão de preço do petróleo utilizando algoritmos de aprendizado de máquina supervisionado e redes neurais (LSTM), bem como disponibilizar uma interface interativa em Streamlit para análise e visualização dos resultados.

    ## Base de Dados
    A base de dados principal foi obtida no repositório do IPEA, com preços mensais do petróleo tipo Brent. A base de eventos históricos foi construída manualmente com base em acontecimentos relevantes (guerras, crises econômicas, decisões da OPEP, etc.).

    ## Pré-processamento
    - Conversão de datas
    - Normalização para modelos de deep learning
    - Criação de janela deslizante para LSTM
    - Separação entre dados de treino e teste (últimos 5 anos para treino)

    ## Modelos Utilizados
    - Regressão Linear
    - Árvore de Decisão
    - Floresta Aleatória (Random Forest)
    - LSTM (Long Short-Term Memory)

    ## Diagrama Geral do Projeto
    """)

    st.image("data/Diagrama.jpg", caption="Fluxo Geral do Projeto")

    st.markdown("""
    ## Resultados
    O modelo LSTM se destacou como o melhor em termos de erro quadrático médio (MSE), erro absoluto médio (MAE) e coeficiente de determinação (R²), conforme apresentado na aba 'Modelos'.

    ## Conclusão
    A utilização de modelos preditivos combinada com uma interface interativa permite não só prever valores futuros do petróleo, mas também entender o impacto de eventos passados. O projeto pode ser expandido futuramente com integração de mais fontes externas e atualização automática da base.
    """)

# -------------------------------------------------------------------
# 📊 MODELOS – avaliação & comparação
# -------------------------------------------------------------------
with tabs[1]:
    st.title("📊 Avaliação de Modelos")

    st.subheader("Métricas")
    metricas_modelos = []
    lstm_pred, window = None, 10

    for nome, modelo in modelos.items():
        if nome != "LSTM" and modelo is not None:
            y_pred = modelo.predict(dias)
            metricas_modelos.append([
                nome,
                round(mean_squared_error(df["preco"], y_pred), 2),
                round(mean_absolute_error(df["preco"], y_pred), 2),
                round(r2_score(df["preco"], y_pred), 2),
            ])

    try:
        lstm = carregar_lstm()
        scaler = carregar_scaler()
        serie = df["preco"].values.reshape(-1, 1)
        serie_scaled = scaler.transform(serie)
        X_lstm = np.array([serie_scaled[i - window : i] for i in range(window, len(serie_scaled))])
        y_lstm_true = serie[window:].flatten()
        y_lstm_pred = scaler.inverse_transform(lstm.predict(X_lstm, verbose=0)).flatten()
        lstm_pred = y_lstm_pred
        metricas_modelos.append([
            "LSTM (melhor modelo)",
            round(mean_squared_error(y_lstm_true, y_lstm_pred), 2),
            round(mean_absolute_error(y_lstm_true, y_lstm_pred), 2),
            round(r2_score(y_lstm_true, y_lstm_pred), 2),
        ])
    except Exception:
        pass

    df_metricas = pd.DataFrame(metricas_modelos, columns=["Modelo", "MSE", "MAE", "R²"])
    st.dataframe(df_metricas, use_container_width=True, hide_index=True)

    st.subheader("Comparar valores reais vs previstos (histórico)")
    comparacao = pd.DataFrame({"Data": df["data"].dt.strftime("%d/%m/%y"), "Real": df["preco"].round(2)})

    for nome, modelo in modelos.items():
        if nome != "LSTM" and modelo is not None:
            comparacao[nome] = np.round(modelo.predict(dias), 2)

    if lstm_pred is not None:
        comparacao["LSTM"] = [np.nan]*window + list(np.round(lstm_pred, 2))

    st.dataframe(comparacao.tail(10), use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# 🔮 PREVISÃO
# -------------------------------------------------------------------
with tabs[2]:
    st.title("🔮 Previsão do Preço do Petróleo Brent")
    st.markdown("""
        <style>
        .blinking {
            animation: blinker 1s linear infinite;
            color: green;
            font-size: 16px;
            font-weight: bold;
        }
        @keyframes blinker {
            50% { opacity: 0; }
        }
        </style>
        <span class="blinking">
            <strong>Observação:</strong> o modelo <strong>LSTM</strong> apresentou o melhor desempenho nas métricas.
        </span>
        """, unsafe_allow_html=True)
    
    modelo_opcao = st.selectbox(
        "Escolha o modelo para previsão:", list(modelos.keys())
    )
    
    dias_previsao = st.slider("Dias a prever:", 1, 10, 5)

    if st.button("Prever"):
        try:
            if modelo_opcao == "LSTM":
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
            else:
                modelo = modelos[modelo_opcao]
                ultimo_dia = dias[-1][0]
                dias_futuros = np.array([[ultimo_dia + i] for i in range(1, dias_previsao + 1)])
                previsoes = modelo.predict(dias_futuros)

            datas_futuras = [df["data"].max() + timedelta(days=i) for i in range(1, dias_previsao + 1)]
            df_previsao = pd.DataFrame({
                "Data": [d.strftime("%d/%m/%y") for d in datas_futuras],
                "Valor Previsto": np.round(previsoes, 2)
            })
            st.success(f"Previsão realizada a partir de {df['data'].max().strftime('%d/%m/%Y')}.")
            st.dataframe(df_previsao, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")

# -------------------------------------------------------------------
# 📅 EVENTOS HISTÓRICOS
# -------------------------------------------------------------------
with tabs[3]:
    st.title("📅 Eventos Históricos vs Preço")

    df_eventos["evento_dropdown"] = df_eventos["data"].dt.strftime("%d/%m/%Y") + " – " + df_eventos["evento"]
    evento_label = st.selectbox("Selecione um evento:", df_eventos["evento_dropdown"].tolist())
    evento_row = df_eventos[df_eventos["evento_dropdown"] == evento_label].iloc[0]
    evento_data = evento_row["data"]
    descricao = evento_row.get("descricao", "")
    janela = st.slider("Dias antes e depois do evento", 5, 60, 15)

    df_plot = df[(df["data"] >= evento_data - pd.Timedelta(days=janela)) &
                 (df["data"] <= evento_data + pd.Timedelta(days=janela))].copy()

    if df_plot.empty:
        st.warning("Não há dados históricos suficientes nesse intervalo para exibir.")
    else:
        df_plot["Data"] = df_plot["data"].dt.strftime("%d/%m/%Y")
        df_plot["Preço"] = df_plot["preco"].round(2)

        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.plot(df_plot["data"], df_plot["Preço"], marker="o")
        ax.axvline(evento_data, color="red", linestyle="--", label="Evento")
        ax.set_title(f"Preço do Petróleo • Evento: {evento_row['evento']}", fontsize=10)
        #ax.set_xlabel("Data", fontsize=8)
        ax.set_ylabel("Preço (US$)", fontsize=8)
        ax.legend()
        plt.xticks(rotation=45, fontsize=7)
        plt.yticks(fontsize=7)
        st.pyplot(fig)

        if descricao:
            st.markdown(f"**Resumo do Evento:** {descricao}")

        st.dataframe(df_plot[["Data", "Preço"]], use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# 👨‍💻 TIME RESPONSÁVEL
# -------------------------------------------------------------------
with tabs[4]:
    st.title("👨‍💻 Time Responsável")
    st.markdown(
        """
        - Ozir José Azevedo Junior 
        - Paloma Cristina Pinheiro
        - Rafael Curti Barros
        - Rilciane de Sousa Bezerra
        """
    )
