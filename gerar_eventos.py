import os
import pandas as pd

eventos = [

    {
        "data": "1990-08-02",
        "evento": "Invasão do Kuwait pelo Iraque",
        "descricao": "O preço quase dobrou, de US$ 17 para US$ 36, chegando a US$ 41,90. A invasão causou incêndios em poços e risco de escassez no Golfo Pérsico."
    },
    {
        "data": "1991-01-17",
        "evento": "Início da Guerra do Golfo",
        "descricao": "O início dos bombardeios acelerou a volatilidade dos preços, mas a rápida resolução do conflito permitiu queda posterior, aliviando temores de interrupção prolongada na oferta."
    },
    {
        "data": "1997-07-02",
        "evento": "Crise financeira asiática",
        "descricao": "A crise reduziu fortemente a demanda por energia na Ásia, derrubando os preços do petróleo em todo o mundo devido à desaceleração econômica regional."
    },
    {
        "data": "2001-09-11",
        "evento": "Atentados terroristas nos EUA",
        "descricao": "Houve uma queda inicial de quase 25% nos preços, seguida de forte volatilidade. O temor de recessão global afetou a demanda, mas a instabilidade geopolítica trouxe incertezas."
    },
    {
        "data": "2003-03-20",
        "evento": "Início da Guerra do Iraque",
        "descricao": "O início da guerra causou alta nos preços devido ao risco de interrupção da produção e aumento da instabilidade na região, que é estratégica para o fornecimento global."
    },
    {
        "data": "2008-09-15",
        "evento": "Falência do Lehman Brothers",
        "descricao": "O preço do barril chegou a US$ 147 antes de despencar para US$ 40. Houve uma bolha especulativa seguida por forte queda da demanda global com a recessão."
    },
    {
        "data": "2014-06-05",
        "evento": "Início da queda dos preços do petróleo",
        "descricao": "Com a explosão da produção de shale oil nos EUA e decisão da OPEP de não cortar produção, o preço caiu de US$ 100 para menos de US$ 50 em poucos meses."
    },
    {
        "data": "2016-01-16",
        "evento": "Suspensão de sanções ao Irã",
        "descricao": "A retirada das sanções permitiu o aumento das exportações de petróleo iraniano, ampliando o excesso de oferta e pressionando os preços para baixo."
    },
    {
        "data": "2020-03-09",
        "evento": "Crise do petróleo por disputa entre Rússia e Arábia Saudita",
        "descricao": "Em meio à COVID-19, a falta de acordo na OPEP+ fez a Arábia Saudita aumentar a produção. Isso gerou a maior queda diária desde 1991, com colapso nos preços."
    },
    {
        "data": "2020-04-20",
        "evento": "Preço do petróleo WTI torna-se negativo pela 1ª vez",
        "descricao": "Pela 1ª vez, o barril WTI ficou negativo: -US$ 37,63. Estoques cheios, queda na demanda e falta de armazenamento foram as causas durante a pandemia."
    }

]

df = pd.DataFrame(eventos)
df["data"] = pd.to_datetime(df["data"])

os.makedirs("data", exist_ok=True)
df.to_csv("data/eventos_petroleo.csv", index=False)
print("Arquivo data/eventos_petroleo.csv criado com sucesso.")
