#Importação das bibliotecas
import streamlit as st 
import pandas as pd
from atualizacao_dados import extrai_dados_ipea_incremental
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from joblib import load
import plotly.express as px


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
.custom-hr {
    border: 0;
    height: 1px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
}
.small-text {
    font-size: 14px;
    color: #666666;
}
.stNumberInput > div {
    padding: 5px;
    width: 100px;  /* Ajuste o valor conforme necessário */
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

st.header('Visão Geral:')

col1, col2, col3 = st.columns(3)
col1.markdown(f'<div class="custom-card"><span class="value">{data_atualizacao}</span><br><span class="label">Dados atualizados até</span></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="custom-card"><span class="value">US$ {preco_minimo}</span><br><span class="label">Menor Preço, registrado em: {data_preco_minimo}</span></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="custom-card"><span class="value">US$ {preco_maximo}</span><br><span class="label">Maior Preço, registrado em: {data_preco_maximo}</span></div>', unsafe_allow_html=True)

# Botão para atualizar os dados
if st.button('Clique para atualizar os dados'):
    dados = extrai_dados_ipea_incremental(url)
    st.success('Dados atualizados!')

# ------------------------------------------------------------------------

# BLOCO 2: STORYTELLING DA CRIAÇÃO DO MODELO COM EXPLICAÇÕES GEOPOLÍTICAS
st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
st.write('# 2. Interpretação Geopolítica na Análise Temporal')

# Texto de contexto
st.markdown("""
<p class="small-text">
Para desenvolver um modelo eficaz de previsão de séries temporais, é crucial compreender o comportamento intrínseco da série em análise. Isso envolve a identificação e a quantificação detalhada de componentes chave que influenciam suas flutuações, como 
sazonalidade, tendência e ruídos. Com esses parâmetros sendo analisados, podemos escolher um modelo mais preciso nas previsões.
</p>
""", unsafe_allow_html=True)


#Criando uma coluna de ano para ajustar o slider
df_eda = dados[['data', 'preco_petroleo']]
df_eda.sort_values(by='data', ascending=True, inplace=True)
# Configurações do Streamlit
st.markdown('## 2.1. Análise da Série Temporal do Preço do Petróleo')

min_date = pd.to_datetime(dados['data'], dayfirst=True).dt.year.min()
max_date = pd.to_datetime(dados['data'], dayfirst=True).dt.year.max()

# Slider para selecionar o intervalo de datas
date_range = st.slider(
    'SELECIONE O INTERVALO QUE DESEJA VISUALIZAR O COMPORTAMENTO DA SÉRIE TEMPORAL',
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# Filtra o DataFrame com base no intervalo selecionado
filtered_df = df_eda[(pd.to_datetime(dados['data'], dayfirst=True).dt.year >= date_range[0]) & (pd.to_datetime(dados['data'], dayfirst=True).dt.year <= date_range[1])]

# Cria o gráfico com Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(x=filtered_df['data'], y=filtered_df['preco_petroleo'], name='Preço Petróleo',line=dict(color='blue')))

fig.update_layout(
   title='Série Temporal do Preço do Petróleo',
    xaxis_title='Data',
    yaxis_title='Preço',
    legend=dict(x=0, y=1.0),
    autosize=True)


# Mostra o gráfico no Streamlit
st.plotly_chart(fig, use_container_width=True)

st.write('### Interpretação:')
st.markdown("""
:blue-background[**1. Pico em 2008:**] O preço do petróleo apresentava um aumento contínuo desde 2002. Do fim de 2007 até a metade de 2008, a proporção desse aumento de preço aumentou substancialmente, chegando a ultrapassar a barreira de U$140. O aumento escalar da época foi causado por um conjunto de fatores:

- Especulação;
            
- Tensões geopolíticas;
            
- Relação entre demanda/oferta. Demanda puxada pelos países emergentes (China e Índia, especialmente.);
            
- O fato de ser um combustível fóssil não renovável;
            
- Mercado financeiro passava por uma tendência de investimento em matérias primas.

:blue-background[**2. Queda abrupta em 2009:**] A famosa crise financeira global de 2009 foi a principal responsável pela queda abrupta nos preços dos barris de petróleo. Como os países estavam em recessão, a demanda por petróleo consequentemente diminuiu no período levando junto o seu preço.

:blue-background[**3. Nova queda abrupta em 2014:**] Causada principalmente pelo aumento da oferta, especulação do mercado, fatores políticos e ambientais.

I) *Aumento da oferta:*

- O EUA investiu em tecnologias inovadoras e aumentou em 40% a sua produção, tornando-se o maior produtor mundial. Os 40% representavam 4 milhões de barris de petróleo por dia, que equivaliam à produção conjunta da Nigéria, Angola e Líbia, alguns dos maiores produtores no país africano. Além disso, isso fez com que os EUA dependessem minimamente da importação do combustível, passando também a exportar (Ou seja, aumentando a oferta).

- O Iraque foi o segundo país com maior aumento de produção no período, cerca de 1 milhão de barris a mais por dia.

- Entrada do Irã no acordo nuclear juntamente ao Grupo 5 +1 (Estados Unidos, Reino Unido, Rússia, China, França, mais Alemanha) possibilitou a volta do Irã ao mercado petroleiro. Na prática a produção aumentou em cerca de mais 300 mil barris por dia.

- O Brasil aumentou significativamente o volume de produção, tornando-se líder na exploração offshore em águas ultraprofundas. Grandes descobertas de petróleo no pré-sal foram descobertas e aumentaram a produção em 400 mil barris por dia.

- A Arábia Saudita devido às suas vastas reservas de petróleo é capaz de influenciar o mercado estrategicamente. Optaram por produzir mais para, segundo analistas, manterem a quota do mercado. Com preços mais baixos e os investimentos em fracking (EUA) e águas ultraprofundas (Brasil) deixam de ser rentáveis. O que ao longo prazo pode tornar a Arábia Saudita a protagonista da exportação.

II) *Especulação do mercado:*

- Muitos analistas e investidores temiam que, por trás dos números oficiais disponibilizados pelo país asiático, se escondia uma realidade bem pior. A queda das ações nas bolsas chinesas foi um indício de que o milagre econômico chinês podia estar próximo ao fim. Isso causou nervosismo nos mercados, pois a China foi um dos principais players que fomentaram o boom na importação dos recursos naturais dos países emergentes, aumentando nos últimos dez anos o seu consumo em mais de 4 milhões de barris por dia, o que representa uma parcela substancial do mercado e com a influência necessária para impactar o preço do barril.

III) *Fatores ambientais:* O ano de 2015 foi um dos mais quente desde que começaram os registros de temperatura no século 19 e o fenômeno meteorológico "El Niño" poderia piorar a situação em 2016. Isso reduziu uma parcela significativa pela busca por gasóleo para aquecimento nos EUA, Europa e Japão. (Ou seja, diminuindo a demanda)

IV) *Fatores políticos:* Apesar dos 13 países-membros da OPEP representarem um terço da produção global de barris de petróleo, nenhum deles cortou a produção no período.

:blue-background[**4. Nova queda abrupta em 2020:**] Impulsionada pelo avanço da pandemia do Covid-19, outro agravante que levou a esse cenário foi o corte nos preços do barril de petróleo pela Arábia Saudita. Isso aconteceu após a ruptura de aliança entre Rússia e a OPEP.

- A epidemia de Covid-19 na China se transformou em uma pandemia global. Isso causou uma recessão na economia de vários países e, por consequência, a redução na demanda pelo combustível.

- Desde 2016, a OPEP já tinha criado uma estratégia de redução na produção (oferta) de barris de petróleo com o objetivo de recuperar os preços dos barris. Isso funcionou até 2020, quando a Rússia se opôs a uma nova proposta de redução. Isso iniciou uma "Guerra de Preços" entre os dois países, com a Arábia Saudita praticando descontos de até U$8, potencializando a queda no valor do commodity drasticamente.

:blue-background[**5. Novo pico em 2022:**] O fim das restrições para o combate à Covid-19 juntamente ao avanço das campanhas de vacinação mundiais melhorou a expectativa pela recuperação da economia mundial, aumentando a confiança dos investidores nos mercados de commodities. A OPEP também coordenou novo corte de produção, diminuindo a oferta do combustível no mercado e, por consequência, alterando a balança comercial.
""")

st.markdown('## 2.2. Fazendo a decomposição da série')

# Preenchendo os dias faltantes (FDS + Feriados)
df_eda.set_index('data', inplace=True)
df_eda = df_eda.asfreq('D').fillna(method='ffill')

# Decompondo a série temporal
resultados = seasonal_decompose(df_eda)
# Criando os gráficos com Plotly
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1)

fig.add_trace(go.Scatter(x=resultados.observed.index, y=resultados.observed, name='Série Original', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=resultados.trend.index, y=resultados.trend, name='Tendência', line=dict(color='purple')), row=2, col=1)
fig.add_trace(go.Scatter(x=resultados.seasonal.index, y=resultados.seasonal, name='Sazonalidade', line=dict(color='orange')), row=3, col=1)
fig.add_trace(go.Scatter(x=resultados.resid.index, y=resultados.resid, name='Resíduo', line=dict(color='rgb(250, 128, 114)') ), row=4, col=1)

# Atualizando o layout para fundo cinza
fig.update_layout(
    height=800, width=800,
    plot_bgcolor='lightgray',
    paper_bgcolor='lightgray',
    title_text="Decomposição da Série Temporal",
    autosize = True
)

# Ajustando títulos dos eixos
fig.update_xaxes(title_text="Data", row=4, col=1)
fig.update_yaxes(title_text="Valor", row=1, col=1)
fig.update_yaxes(title_text="Valor", row=2, col=1)
fig.update_yaxes(title_text="Valor", row=3, col=1)
fig.update_yaxes(title_text="Valor", row=4, col=1)

# Exibindo o gráfico no Streamlit
st.plotly_chart(fig, use_container_width=True)


st.markdown("""
Analisando a decomposição da série temporal, podemos observar que:
            
**- Tendência:** A tendência geral ao longo do tempo indica um crescimento nos preços do petróleo. Apesar das flutuações e dos períodos de queda, a direção geral parece ser ascendente.

**- Sazonalidade:** Não há um padrão sazonal forte ou significativo, indicando que os preços do petróleo não seguem um ciclo repetitivo regular.
            
**- Ruído:** O ruído mostra a variabilidade aleatória e os eventos imprevisíveis que afetam os preços do petróleo.
""")

# -------------------------------------------------------------------------------------
# BLOCO 3: MODELO E PERFORMANCE
st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
st.write('# 3. Construindo a predição')
st.write('## 3.1. Verificando a estacionariedade')

st.markdown("""
Para selecionar o modelo de machine learning a ser usado nesse projeto, foi feito verificação da estacionariedade atrav~es do **Teste de Dickey-Fuller Aumentado (ADF)**
""")
adf_result = adfuller(df_eda['preco_petroleo'])

# Exibindo os resultados no Streamlit
st.write('#### Resultados encontrados')
st.write(f"###### **Teste Estatístico:** {adf_result[0]}")
st.write(f"###### **Valor-p:** {adf_result[1]}")
st.write("###### **Valores Críticos:**")
for key, value in adf_result[4].items():
    st.write(f"   {key}: {value}")

if adf_result[1] > 0.05:
    st.warning("A série temporal não é estacionária (p-valor > 0.05).")

st.markdown("""
Como o valor de p-value está muito alto e o valor do Teste Estatístico é muito maior do que os valores críticos, concluímos que a **série temporal não é estacionária**.
            
Por isso, o modelo Prophet foi entendido como adequado para a solução.
            """)

st.write('## 3.2. Treinando o Modelo')

st.markdown("""
Incialmente, treinou-se o modelo utilizando a série temporal inteira, e obteve-se um MAPE igual a **17.84%**, o que não considera-se adequado.
            
Decidiu-se então utilizar os últimos 365 dias de dados para treinar o modelo e o MAPE reduziu para **1.94%**.
            
Para mais informações sobre parâmetros de treinamento utilizados, [clique aqui](https://github.com/sergiolvelloso/grupo119-dtat-brent-price/blob/main/dev-modelo.ipynb).
            """)


# Criando e ajustando o modelo Prophet
df_prophet = df_eda.reset_index().rename(columns={'data':'ds', 'preco_petroleo':'y'})
df_prophet_um_ano = df_prophet[-365:]

#Separando em base de treino e teste
split_point = int(len(df_prophet_um_ano) * 0.9)
treino_um_ano = df_prophet_um_ano.iloc[:split_point]
valid_um_ano = df_prophet_um_ano.iloc[split_point:]
print(f'Treino : {treino_um_ano.shape}')
print(f'Valid: {valid_um_ano.shape}')

modelo = Prophet(daily_seasonality=True)
modelo.fit(treino_um_ano)
dataFramefuture = modelo.make_future_dataframe(periods=7, freq='D')
previsao_um_ano = modelo.predict(dataFramefuture)

# Plotando usando Plotly
fig = go.Figure()

# Adicionando a série original
fig.add_trace(go.Scatter(x=treino_um_ano['ds'], y=treino_um_ano['y'],
                         mode='markers', name='Dados de Treinamento', marker=dict(color='red')))

# Adicionando a previsão
fig.add_trace(go.Scatter(x=previsao_um_ano['ds'], y=previsao_um_ano['yhat'],
                         mode='lines', name='Previsão', line=dict(color='blue')))

# Adicionando intervalo de incerteza
fig.add_trace(go.Scatter(x=previsao_um_ano['ds'], y=previsao_um_ano['yhat_upper'],
                         mode='lines', name='Intervalo Superior', line=dict(color='lightblue'),
                         fill='tonexty'))
fig.add_trace(go.Scatter(x=previsao_um_ano['ds'], y=previsao_um_ano['yhat_lower'],
                         mode='lines', name='Intervalo Inferior', line=dict(color='lightblue'),
                         fill='tonexty'))

# Ajustando o layout
fig.update_layout(
    title='Previsão com Prophet',
    xaxis_title='Data',
    yaxis_title='Preço',
    plot_bgcolor='lightgray',
    paper_bgcolor='lightgray',
)

# Exibindo o gráfico no Streamlit
st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------
# BLOCO 4: Deploy do modelo
st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
st.write('# 4. Fazer a Previsão do Preço do Petróleo')

# Carregar o modelo treinado
modelo = load('modelo_prophet.joblib')

def fazer_previsao(periods):
    future = modelo.make_future_dataframe(periods=periods, freq='D')
    forecast = modelo.predict(future)
    return forecast

periods = st.number_input( 'Dias a serem previstos', min_value=1, max_value=7, value=7)


if st.button('Fazer Previsão'):
    forecast = fazer_previsao(periods)

    # Mostrar a tabela de previsões
    st.subheader('Previsões')
    forecast['ds'] = forecast['ds'].dt.strftime('%d/%m/%Y')
    forecast = forecast.rename(columns={
                                    'ds': 'Data',
                                    'yhat': 'Previsão'
                                })
    st.write(forecast[['Data', 'Previsão']].tail(periods))

    # Plotar as previsões para apenas o período previsto
    forecast_subset = forecast.tail(periods)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_subset['Data'], y=forecast_subset['Previsão'], mode='lines', name='Previsão'))
    fig.update_layout(
        title=f'Previsões dos próximos {periods} dias',
        xaxis_title='Data',
        yaxis_title='Preço do Petróleo'
    )
    st.plotly_chart(fig, use_container_width=True)
    # Componentes do modelo para apenas o período previsto
    fig_trend = px.line(forecast_subset, x='Data', y='trend', title='Trend Component')
    fig_weekly = px.line(forecast_subset, x='Data', y='weekly', title='Weekly Component')
    st.plotly_chart(fig_trend, use_container_width=True)
    st.plotly_chart(fig_weekly, use_container_width=True)

# BLOCO 4: FINALIZAÇÃO
st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
st.write('# 5. Links Úteis')

st.markdown('[Repositório no Github](https://github.com/sergiolvelloso/grupo119-dtat-brent-price)')
st.markdown('[Dados Históricos do Preço do Petróleo Brent](http://www.ipeadata.gov.br/)')
st.markdown('[Documentação do Prophet](https://facebook.github.io/prophet/)')
st.markdown('[Documentação do Plotly](https://plotly.com/python/)')

st.write('#### Equipe 119')
st.write('##### Bruna Batista do Carmo Abreu - RM 351370')
st.write('##### Sérgio Luiz Velloso Filho - RM 351371')











