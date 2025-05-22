# treinar_modelos.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ============ Diretórios ============
os.makedirs("modelos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ============ Carregar dados ============
df = pd.read_excel("data/IPEA_DB.xlsx")
df.columns = ["data", "preco"]
df = df.sort_values("data")
df.reset_index(drop=True, inplace=True)

# ============ Filtro: últimos 5 anos para treino ============
ultimo_ano = df["data"].max()
limite_data = ultimo_ano - pd.DateOffset(years=5)
df_treino = df[df["data"] >= limite_data].copy()
df_treino["dias"] = (df_treino["data"] - df_treino["data"].min()).dt.days

X = df_treino[["dias"]].values
y = df_treino["preco"].values

# ============ Modelos convencionais ============
modelos = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor()
}

metricas = []
predicoes = {}

for nome, modelo in modelos.items():
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    metricas.append({"modelo": nome, "MSE": mse, "MAE": mae, "R2": r2})
    predicoes[nome] = y_pred

# Melhor modelo
metricas_df = pd.DataFrame(metricas)
melhor_modelo_nome = metricas_df.sort_values("R2", ascending=False).iloc[0]["modelo"]
melhor_modelo = modelos[melhor_modelo_nome]
with open("modelos/melhor_modelo.pkl", "wb") as f:
    pickle.dump(melhor_modelo, f)

# ============ Modelo LSTM ============
window = 10
scaler = MinMaxScaler()
serie = df_treino["preco"].values.reshape(-1, 1)
serie_scaled = scaler.fit_transform(serie)

X_lstm, y_lstm = [], []
for i in range(window, len(serie_scaled)):
    X_lstm.append(serie_scaled[i - window:i])
    y_lstm.append(serie_scaled[i])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(window, 1)),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm, y_lstm, epochs=50, batch_size=16, verbose=0, callbacks=[EarlyStopping(patience=5)])

model_lstm.save("modelos/melhor_modelo_lstm", save_format="tf")
with open("modelos/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ============ Resultados ============
metricas_df.to_csv("resultados/metricas_modelos.csv", index=False)

plt.figure(figsize=(10, 4))
plt.plot(df_treino["data"], y, label="Real")
for nome, y_pred in predicoes.items():
    plt.plot(df_treino["data"], y_pred, label=nome)
plt.title("Comparação de Modelos Convencionais (últimos 5 anos)")
plt.xlabel("Data")
plt.ylabel("Preço")
plt.legend()
plt.tight_layout()
plt.savefig("resultados/grafico_comparativo.png")

print("✅ Modelos treinados com dados dos últimos 5 anos.")
