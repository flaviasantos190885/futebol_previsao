# -*- coding: utf-8 -*-
"""
Script de Automação do Pipeline de Machine Learning para Futebol

Este script automatiza as etapas de ETL (Extração, Transformação, Carga)
e o treinamento/salvamento do modelo de ML para prever resultados de futebol.
Pode ser agendado para rodar periodicamente (ex: via Agendador de Tarefas do Windows).
"""

import pandas as pd
import numpy as np
import re
import unicodedata
import sqlite3
import os
from pycaret.classification import setup, create_model, tune_model, finalize_model, save_model, compare_models
import random # Importar para geração de dados randômicos
from datetime import datetime, timedelta # Importar para datas randômicas

# --- Configurações de Caminho ---
# Defina o caminho base do seu projeto.
# Certifique-se de que este caminho está correto no seu sistema Windows.
# Exemplo: base_path = "C:\\Users\\SeuUsuario\\Downloads\\Trabalho_Final_VIZ"
base_path = os.path.dirname(os.path.abspath(__file__)) # Pega o diretório onde este script está

data_path = os.path.join(base_path, "data")
model_path = os.path.join(base_path, "model")

# Certifique-se de que as pastas existem
os.makedirs(data_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Caminhos dos arquivos de banco de dados e CSV
db_producao_path = os.path.join(data_path, "futebol_prod.db")
db_futebol_dw_path = os.path.join(data_path, "futebol_dw.db")
# Este CSV será usado como base para obter valores únicos para dados randômicos
source_csv_for_extract = os.path.join(data_path, "BRA_brasileirao_final.csv")
# Este CSV será o resultado final do tratamento para debug/verificação
treated_csv_output = os.path.join(data_path, "BRA_brasileirao_tratado_final.csv")


# --- Funções de Pré-processamento (Reutilizadas do seu código de treinamento) ---
def slugify(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return text

def create_result_column(df):
    if "resultado" not in df.columns:
        if "gols_mandante" in df.columns and "gols_visitante" in df.columns:
            df["gols_mandante"] = pd.to_numeric(df["gols_mandante"], errors="coerce").fillna(0).astype(int)
            df["gols_visitante"] = pd.to_numeric(df["gols_visitante"], errors="coerce").fillna(0).astype(int)
            df["resultado"] = df.apply(
                lambda row: "Casa" if row["gols_mandante"] > row["gols_visitante"]
                else "Fora" if row["gols_mandante"] < row["gols_visitante"]
                else "Empate",
                axis=1
            )
        else:
            raise KeyError("As colunas 'gols_mandante' ou 'gols_visitante' não foram encontradas para gerar a coluna 'resultado'.")
    return df[df["resultado"].isin(["Casa", "Fora", "Empate"])].copy()

def apply_desempate_logic(df):
    vitorias_casa = df[df['resultado'] == 'Casa'].groupby('time_mandante').size()
    vitorias_fora = df[df['resultado'] == 'Fora'].groupby('time_visitante').size()

    def desempate(row):
        if row['resultado'] == 'Empate':
            mandante = row['time_mandante']
            visitante = row['time_visitante']
            casa_wins = vitorias_casa.get(mandante, 0)
            fora_wins = vitorias_fora.get(visitante, 0)
            
            if casa_wins > fora_wins:
                return 'Casa'
            elif fora_wins > casa_wins:
                return 'Fora'
            else:
                return 'Empate'
        else:
            return row['resultado']
    df['resultado_ajustado'] = df.apply(desempate, axis=1)
    return df

def handle_date_columns(df):
    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce", dayfirst=True)
        df.dropna(subset=['data'], inplace=True)
        df["ano"] = df["data"].dt.year.astype(int)
        df["mes"] = df["data"].dt.month.astype(int)
    else:
        print("AVISO: Coluna 'data' não encontrada. Usando ano=2023 e mes=1 como padrão.")
        df["ano"] = 2023
        df["mes"] = 1
    return df

def handle_outliers(df, num_cols_for_outliers):
    for col in num_cols_for_outliers:
        if col in df.columns and df[col].nunique() > 1 and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
            temp_col = df[col].dropna()
            if not temp_col.empty:
                q1, q3 = temp_col.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    df[col] = np.clip(df[col], lower, upper)
                else:
                    print(f"AVISO: IQR inválido (zero ou NaN) para a coluna '{col}'. Outliers não foram tratados.")
            else:
                print(f"AVISO: Coluna '{col}' está vazia após conversão para numérico. Não foi possível tratar outliers.")
        elif col in df.columns:
            print(f"AVISO: Coluna '{col}' não tratada para outliers (não numérica ou apenas um valor único).")
        else:
            print(f"AVISO: Coluna '{col}' não encontrada para tratamento de outliers.")
    return df

def handle_categorical_nans(df, categorical_cols):
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(r'^\s*$', np.nan, regex=True)
            if df[col].isnull().any():
                if not df[col].mode().empty:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna('Desconhecido', inplace=True)
        else:
            print(f"AVISO: Coluna categórica '{col}' não encontrada para tratamento de valores ausentes.")
    return df


# --- Etapas do Pipeline ---

def _extract():
    print("\n=============== Etapa de Extração ===============\n")
    # Simulação de Extração: Gerar dados randômicos e carregar em um banco de dados SQLite de produção.
    
    try:
        # Carrega o CSV existente para obter a estrutura e valores únicos
        df_base = pd.read_csv(source_csv_for_extract)
        df_base.columns = [slugify(col) for col in df_base.columns]

        # --- Geração de Dados Randômicos ---
        num_random_rows = 5 # Quantidade de novas linhas de dados a serem geradas

        # Obter valores únicos para colunas categóricas da base existente
        unique_teams = pd.concat([df_base['time_mandante'], df_base['time_visitante']]).dropna().unique().tolist()
        unique_estadios = df_base['estadio'].dropna().unique().tolist()
        unique_arbitros = df_base['arbitro'].dropna().unique().tolist()
        unique_tecnicos = pd.concat([df_base['tecnico_mandante'], df_base['tecnico_visitante']]).dropna().unique().tolist()
        
        # Se alguma lista estiver vazia, preencha com valores padrão para evitar erros
        if not unique_teams: unique_teams = ['Time A', 'Time B']
        if not unique_estadios: unique_estadios = ['Estadio Padrao']
        if not unique_arbitros: unique_arbitros = ['Arbitro Padrao']
        if not unique_tecnicos: unique_tecnicos = ['Tecnico Padrao']


        random_data_list = []
        for _ in range(num_random_rows):
            mandante = random.choice(unique_teams)
            visitante = random.choice([t for t in unique_teams if t != mandante]) # Garante times diferentes
            
            gols_mandante = random.randint(0, 5)
            gols_visitante = random.randint(0, 5)

            if gols_mandante > gols_visitante:
                resultado = "Casa"
            elif gols_mandante < gols_visitante:
                resultado = "Fora"
            else:
                resultado = "Empate"
            
            # Gerar data randômica para o ano atual ou próximo
            current_year = datetime.now().year
            random_month = random.randint(1, 12)
            random_day = random.randint(1, 28) # Para evitar problemas com meses de 31 dias
            random_date = datetime(current_year, random_month, random_day).strftime('%d/%m/%Y')

            random_row = {
                'ano_campeonato': current_year,
                'data': random_date,
                'rodada': random.randint(1, 38),
                'estadio': random.choice(unique_estadios),
                'arbitro': random.choice(unique_arbitros),
                'publico': random.randint(5000, 50000),
                'publico_max': random.randint(50000, 80000),
                'time_mandante': mandante,
                'time_visitante': visitante,
                'tecnico_mandante': random.choice(unique_tecnicos),
                'tecnico_visitante': random.choice(unique_tecnicos),
                'colocacao_mandante': random.randint(1, 20),
                'colocacao_visitante': random.randint(1, 20),
                'valor_equipe_titular_mandante': round(random.uniform(1000000, 50000000), 2),
                'valor_equipe_titular_visitante': round(random.uniform(1000000, 50000000), 2),
                'idade_media_titular_mandante': round(random.uniform(22, 32), 2),
                'idade_media_titular_visitante': round(random.uniform(22, 32), 2),
                'gols_mandante': gols_mandante,
                'gols_visitante': gols_visitante,
                'gols_1_tempo_mandante': random.randint(0, gols_mandante),
                'gols_1_tempo_visitante': random.randint(0, gols_visitante),
                'escanteios_mandante': random.randint(0, 15),
                'escanteios_visitante': random.randint(0, 15),
                'faltas_mandante': random.randint(5, 25),
                'faltas_visitante': random.randint(5, 25),
                'chutes_bola_parada_mandante': random.randint(0, 10),
                'chutes_bola_parada_visitante': random.randint(0, 10),
                'defesas_mandante': random.randint(0, 10),
                'defesas_visitante': random.randint(0, 10),
                'impedimentos_mandante': random.randint(0, 5),
                'impedimentos_visitante': random.randint(0, 5),
                'chutes_mandante': random.randint(5, 25),
                'chutes_visitante': random.randint(5, 25),
                'chutes_fora_mandante': random.randint(0, 15),
                'chutes_fora_visitante': random.randint(0, 15),
                'resultado': resultado,
                # 'resultado_ajustado', 'ano', 'mes' serão criados na transformação
            }
            random_data_list.append(random_row)

        df_random = pd.DataFrame(random_data_list)
        # Normaliza nomes de colunas para os dados randômicos também
        df_random.columns = [slugify(col) for col in df_random.columns]

        # Conecta ao banco de dados de produção
        conn = sqlite3.connect(db_producao_path)
        # Adiciona os dados randômicos à tabela existente (ou cria se não existir)
        df_random.to_sql("partidas_producao", conn, if_exists="append", index=False)
        conn.close()
        print(f"{num_random_rows} novas linhas de dados randômicos geradas e adicionadas a '{db_producao_path}'.")
        
        # Para verificar o total de dados após a adição
        conn_check = sqlite3.connect(db_producao_path)
        total_rows = pd.read_sql_query("SELECT COUNT(*) FROM partidas_producao", conn_check).iloc[0,0]
        conn_check.close()
        print(f"Total de linhas na tabela 'partidas_producao': {total_rows}")

    except Exception as e:
        print(f"ERRO na etapa de Extração (geração de dados randômicos): {e}")
        raise # Re-lança o erro para parar o pipeline

def _transform():
    print("\n=============== Etapa de Transformação ===============\n")
    # Conecta ao banco de dados de produção e lê os dados
    try:
        conn_prod = sqlite3.connect(db_producao_path)
        # Lendo todos os dados para reprocessar, incluindo os novos randômicos
        df_raw = pd.read_sql_query("SELECT * FROM partidas_producao", conn_prod)
        conn_prod.close()
        print(f"Dados lidos de '{db_producao_path}' para transformação. Shape: {df_raw.shape}")

        df_tratado = df_raw.copy()

        # Aplica as funções de tratamento
        df_tratado = create_result_column(df_tratado)
        df_tratado = apply_desempate_logic(df_tratado)
        df_tratado = handle_date_columns(df_tratado)
        
        # Garante a coluna 'competicao'
        if 'competicao' not in df_tratado.columns:
            df_tratado['competicao'] = 'Brasileirao'

        # Colunas numéricas para tratamento de outliers (ajuste conforme seu dataset)
        num_cols_for_outliers = [
            'publico', 'publico_max', 'colocacao_mandante', 'colocacao_visitante',
            'valor_equipe_titular_mandante', 'valor_equipe_titular_visitante',
            'idade_media_titular_mandante', 'idade_media_titular_visitante',
            'gols_mandante', 'gols_visitante', 'gols_1_tempo_mandante', 'gols_1_tempo_visitante',
            'escanteios_mandante', 'escanteios_visitante', 'faltas_mandante', 'faltas_visitante',
            'chutes_bola_parada_mandante', 'chutes_bola_parada_visitante', 'defesas_mandante', 'defesas_visitante',
            'impedimentos_mandante', 'impedimentos_visitante', 'chutes_mandante', 'chutes_visitante',
            'chutes_fora_mandante', 'chutes_fora_visitante', 'rodada', 'ano', 'mes'
        ]
        df_tratado = handle_outliers(df_tratado, num_cols_for_outliers)

        # Colunas categóricas para tratamento de NaNs (ajuste conforme seu dataset)
        categorical_cols_for_nans = [
            'estadio', 'arbitro', 'time_mandante', 'time_visitante',
            'tecnico_mandante', 'tecnico_visitante', 'competicao'
        ]
        df_tratado = handle_categorical_nans(df_tratado, categorical_cols_for_nans)

        # Conecta ao banco de dados de futebol tratado e carrega os dados transformados
        conn_dw = sqlite3.connect(db_futebol_dw_path)
        df_tratado.to_sql("partidas_tratadas", conn_dw, if_exists="replace", index=False) # Sempre substitui para ter a versão mais recente
        conn_dw.close()
        print(f"Dados transformados e carregados em '{db_futebol_dw_path}'.")
        print(f"Shape dos dados transformados: {df_tratado.shape}")

        # Opcional: Salvar o CSV tratado para verificação
        df_tratado.to_csv(treated_csv_output, index=False)
        print(f"Dados tratados também salvos em '{treated_csv_output}'.")

    except Exception as e:
        print(f"ERRO na etapa de Transformação: {e}")
        raise # Re-lança o erro para parar o pipeline

def _train_and_save_model():
    print("\n=============== Etapa de Treinamento e Salvamento do Modelo ===============\n")
    # Conecta ao banco de dados de futebol tratado e lê os dados
    try:
        conn_dw = sqlite3.connect(db_futebol_dw_path)
        df_model_data = pd.read_sql_query("SELECT * FROM partidas_tratadas", conn_dw)
        conn_dw.close()
        print(f"Dados lidos de '{db_futebol_dw_path}' para treinamento do modelo. Shape: {df_model_data.shape}")

        # Colunas a serem ignoradas pelo PyCaret (ajuste conforme seu setup original)
        ignore_features_list = [
            "data", "gols_mandante", "gols_visitante", "resultado",
            "ano_campeonato", "arbitro", "tecnico_mandante", "tecnico_visitante", "publico_max"
        ]
        ignore_features_existing = [col for col in ignore_features_list if col in df_model_data.columns]

        # Configurar o ambiente do PyCaret para classificação
        exp = setup(
            data=df_model_data,
            target="resultado_ajustado",
            session_id=42,
            fix_imbalance=False,
            remove_outliers=False,
            normalize=True,
            ignore_features=ignore_features_existing,
            verbose=False,
        )
        print("Ambiente PyCaret configurado.")

        # Comparar modelos e selecionar o melhor
        print("Comparando modelos (pode levar alguns minutos)...")
        best_model = compare_models(fold=5, sort='Accuracy')
        print(f"Melhor modelo encontrado: {best_model}")

        # Finalizar e salvar o modelo
        final_model = finalize_model(best_model)
        save_model(final_model, os.path.join(model_path, "modelo-final"))
        print(f"Modelo salvo em '{os.path.join(model_path, 'modelo-final.pkl')}'.")

    except Exception as e:
        print(f"ERRO na etapa de Treinamento e Salvamento do Modelo: {e}")
        raise # Re-lança o erro para parar o pipeline

# --- Execução do Pipeline ---
if __name__ == "__main__":
    try:
        print("Iniciando o pipeline de automação...")
        _extract()
        _transform()
        _train_and_save_model()
        print("\nPipeline de automação concluído com sucesso!")
    except Exception as e:
        print(f"\nO pipeline falhou: {e}")

