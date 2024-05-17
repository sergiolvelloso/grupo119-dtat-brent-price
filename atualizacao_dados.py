import requests
from bs4 import BeautifulSoup
from datetime import datetime

import pandas as pd

#Função para extrair os dados de forma incremental do site IPEA
def extrai_dados_ipea_incremental(url):
    
    resposta = requests.get(url)

    if resposta.status_code == 200:

        soup = BeautifulSoup(resposta.text, 'html.parser')
        table = soup.find('table', {'id': 'grd_DXMainTable'})

        df_new_data = pd.read_html(str(table), header=0)[0]
        df_new_data.columns = ['data', 'preco_petroleo' ]
        df_new_data['data'] = pd.to_datetime(df_new_data['data'], dayfirst=True)
        df_new_data['preco_petroleo'] = df_new_data['preco_petroleo']/100

        path = 'dados\ipea.csv'

        try:
            df_existente = pd.read_csv(path)
            print(df_existente)
            df_existente['data'] = pd.to_datetime(df_existente['data'], format="%Y-%m-%d")
            

        except FileNotFoundError:
            df_existente = df_new_data
        

        
        #Caso já tenha um arquivo salvo, entende de quando deverá puxar os novos dados
        last_date = df_existente['data'].max()
        new_rows = df_new_data[df_new_data['data'] > last_date]

        if new_rows.empty:
            updated_df = df_existente
        else:
            updated_df = pd.concat([df_existente, new_rows], ignore_index=True)
            
        #Gera o arquivo atualizado
        updated_df.to_csv(path, index=False)
        updated_df.head()

        return updated_df

    else:
        print("Falha ao acessar a página: Status Code", resposta.status_code)