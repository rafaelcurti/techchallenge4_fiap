# ⛽ Dashboard Interativo: Previsão e Análise do Petróleo Brent

Este projeto apresenta um dashboard profissional em Streamlit para:

- 📈 Previsão do preço do petróleo Brent com modelos de machine learning
- 🔍 Análise de impacto de eventos históricos nos preços
- 📊 Comparação entre modelos tradicionais e LSTM

---

## 🎯 Objetivo

Avaliar o comportamento do preço do petróleo Brent ao longo do tempo, utilizando técnicas de ciência de dados, e verificar a influência de eventos geopolíticos e econômicos relevantes sobre o mercado.

---

## 🛠 Tecnologias utilizadas

- Python 3.10
- Streamlit
- Pandas / Numpy / Matplotlib
- Scikit-learn
- TensorFlow (LSTM)
- Openpyxl

---

## 📂 Estrutura do projeto

```
├── app.py                      # Streamlit principal
├── treinar_modelos.py         # Treinamento e avaliação de modelos
├── requirements.txt           # Dependências do projeto
├── data/
│   ├── IPEA_DB.xlsx           # Base de dados histórica
│   ├── eventos_petroleo.csv   # Datas e descrições de eventos
│   ├── print_*.png            # Prints para a aba de documentação
├── modelos/
│   ├── melhor_modelo.pkl
│   ├── melhor_modelo_lstm.keras
│   └── scaler.pkl
├── resultados/
│   ├── grafico_comparativo.png
│   └── metricas_modelos.csv
```

---

## 🚀 Como rodar localmente

1. **Crie o ambiente com Anaconda**:
```bash
conda create -n techchallenge4 python=3.10 -y
conda activate techchallenge4
```

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

3. **(Opcional)**: Treine os modelos com:
```bash
python treinar_modelos.py
```

4. **Execute o app**:
```bash
streamlit run app.py
```

---

## 🌐 Como publicar no Streamlit Cloud

1. Suba todos os arquivos para um repositório no GitHub
2. Vá para: [streamlit.io/cloud](https://streamlit.io/cloud)
3. Conecte sua conta GitHub e clique em **"New app"**
4. Selecione:
   - Repositório: `seu_usuario/techchallenge4`
   - Branch: `main`
   - Arquivo principal: `app.py`
5. Clique em **Deploy**

---

## 📸 Capturas do Dashboard

> Inclua aqui prints das abas principais, se desejar.

---

## 👨‍💻 Desenvolvedores

- Ozir José Azevedo Junior 
- Paloma Cristina Pinheiro
- Rafael Curti Barros
- Rilciane de Sousa Bezerra 

---

## 📘 Licença

Este projeto é apenas para fins educacionais (FIAP Tech Challenge).
