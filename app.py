#Importação das bibliotecas
import streamlit as st 
import pandas as pd
from atualizacao_dados import extrai_dados_ipea_incremental


# Configurações Gerais da Página
st.set_page_config(page_title="Modelo Preditivo do Petróleo ", layout="wide")

st.markdown("""
<style>
.custom-card {
    text-align: center;
    margin: 20px;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.15); /* Sombra para dar um efeito elevado */
    font-size: 20px;
    background-color: #333333; /* Cor de fundo dos cartões */
    color: #FFFFFF; /* Cor do texto */
}
.custom-card .value {
    font-size: 30px;
    font-weight: bold;
}
.custom-card .label {
    font-size: 18px;
    color: #BBBBBB; /* Cor do rótulo */
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)


#Persomalização da separação dos blocos
st.markdown("""
<style>
.custom-hr {
    border: 0;
    height: 1px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
}
</style>
""", unsafe_allow_html=True)


# Atualização dos Dados
url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'
dados = extrai_dados_ipea_incremental(url)
dados['data'] = pd.to_datetime(dados['data'], format='%Y-%m-%d').dt.date

## BLOCO 1 - ANALISANDO OS DADOS E INSITGHS QUE PODEMOS TER

# --- Página Inicial
st.write('# 1. Estudo do Preço do Petróleo Brent')

# Texto de contexto
st.markdown("""
<p class="small-text">
Este painel foi desenvolvido para fornecer uma visão clara e atualizada dos movimentos de preço do Petróleo Brent, trazendo insithgs baseados nos dados históricos e aplicando um modelo de previsão dos dados futuros. 
Aqui você encontrará informações essenciais que ajudarão você a entender melhor as tendências de preços e seus possíveis impactos.
</p>
""", unsafe_allow_html=True)

# --- Colocando Big Numbers
data_atualizacao = dados['data'].max().strftime('%d/%m/%Y')
preco_minimo = dados['preco_petroleo'].min()
preco_maximo = dados['preco_petroleo'].max()

df_data_valor_minimo = dados.loc[dados['preco_petroleo'] == preco_minimo]
data_preco_minimo = df_data_valor_minimo['data'].max().strftime('%d/%m/%Y')

df_data_valor_maximo = dados.loc[dados['preco_petroleo'] == preco_maximo]
data_preco_maximo = df_data_valor_maximo['data'].max().strftime('%d/%m/%Y')

st.header('Entenda o contexto geral:')

col1, col2, col3 = st.columns(3)
col1.markdown(f'<div class="custom-card"><span class="value">{data_atualizacao}</span><br><span class="label">Dados atualizados até</span></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="custom-card"><span class="value">US$ {preco_minimo}</span><br><span class="label">Menor Preço, registrado em: {data_preco_minimo}</span></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="custom-card"><span class="value">US$ {preco_maximo}</span><br><span class="label">Maior Preço, registrado em: {data_preco_maximo}</span></div>', unsafe_allow_html=True)

# Botão para atualizar os dados
if st.button('Clique para atualizar os dados'):
    dados = extrai_dados_ipea_incremental(url)
    st.success('Dados atualizados!')

## GRÁFICOS PARA INSIGTHS AQUI

# BLOCO 2: STORYTELLING DA CRIAÇÃO DO MODELO COM EXPLICAÇÕES GEOPOLÍTICAS
st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
st.write('# 2. Interpretação Geopolítica na Análise Temporal')

# Texto de contexto
st.markdown("""
<p class="small-text">
Para desenvolver um modelo eficaz de previsão de séries temporais, é crucial compreender o comportamento intrínseco da série em análise. Isso envolve a identificação e a quantificação detalhada de componentes chave que influenciam suas flutuações, como 
sazonalidade, tendência e ruídos. Com esses parâmetros sendo analisados, podemos construir um modelo mais preciso nas previsões.
</p>
""", unsafe_allow_html=True)


st.header('Fazendo a decomposição da série: ')



# BLOCO 3: MODELO E PERFORMANCE

# BLOCO 4: FINALIZAÇÃO
st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
st.write('# 4. Links Úteis')

st.markdown('[Repositório no Github](https://github.com/sergiolvelloso/grupo119-dtat-brent-price)')

st.write('#### Equipe 119')
st.write('##### Bruna Batista do Carmo Abreu - RM 351370')
st.write('##### Sérgio Luiz Velloso Filho - RM 351371')











