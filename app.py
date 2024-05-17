#Importação das bibliotecas
import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
import utils 
from sklearn.pipeline import Pipeline
import joblib
from joblib import load
from atualizacao_dados import extrai_dados_ipea_incremental


#Carregando os dados
url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'
dados = extrai_dados_ipea_incremental(url)

dados['data'] = pd.to_datetime(dados['data'], format='%Y-%m-%d').dt.date
# --- Página Inicial
st.write('# Estudo do Preço do Petróleo Brent')

# --- Colocando Big Numbers
col1, col2, col3 = st.columns(3)

# Cartão 1: Data de atualização dos dados
col1.metric(label="Dados atualizados até", value=(dados['data'].max().strftime('%d/%m/%Y')))

# Cartão 2: Menor Preço já registrado
col2.metric(label="Menor Preço", value=str(dados['preco_petroleo'].min()))

# Cartão 3: Maior Preço já registrado
col3.metric(label="Maior Preço", value=str(dados['preco_petroleo'].max()))

# Botão para atualizar os dados
if st.button('Clique para atualizar os dados'):
    dados = extrai_dados_ipea_incremental(url)
    st.success('Dados atualizados!')



