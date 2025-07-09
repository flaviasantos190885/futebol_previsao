import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model
import os
import re
import unicodedata
import numpy as np
import base64 # Importar o módulo base64

st.set_page_config(layout="wide", page_title="Resultados do Brasileirão Série A")

# --- Carregar e Codificar Imagem de Fundo para Base64 ---
def get_base64_image(image_path):
    """Lê uma imagem local e a codifica em Base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Erro: Imagem de fundo '{image_path}' não encontrada. Certifique-se de que o caminho está correto.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar ou codificar a imagem de fundo: {e}")
        return None

# Caminho da imagem de fundo
background_image_path = os.path.join("img", "fundo.jpg")
base64_image = get_base64_image(background_image_path)

# Se a imagem foi carregada com sucesso, use-a no CSS
background_style = ""
if base64_image:
    background_style = f"background-image: url('data:image/jpeg;base64,{base64_image}');"
else:
    # Fallback para uma cor de fundo sólida se a imagem não carregar
    background_style = "background-color: #1a1a1a;" # Um verde escuro como fallback
    st.warning("Usando cor de fundo sólida, pois a imagem não pôde ser carregada.")

# # --- Estilos CSS Personalizados (APENAS para o fundo) ---
# st.markdown(f"""
# <style>
# /* Estilo para a área principal do Streamlit (fundo de campo de futebol) */
# [data-testid="stAppViewContainer"] {{
#     {background_style} /* Imagem de campo de futebol na pasta img ou cor de fallback */
#     background-size: cover; /* Cobre toda a área */
#     background-position: center; /* Centraliza a imagem */
#     background-attachment: fixed; /* Mantém o fundo fixo ao rolar */
# }}
# </style>
# """, unsafe_allow_html=True)

st.markdown(f"""
<style>
/* Estilo para a área principal do Streamlit (fundo de campo de futebol) */
[data-testid="stAppViewContainer"] {{
    {background_style} /* Imagem de campo de futebol na pasta img ou cor de fallback */
    background-size: cover; /* Cobre toda a área */
    background-position: center; /* Centraliza a imagem */
    background-attachment: fixed; /* Mantém o fundo fixo ao rolar */
}}

/* Estilo para a barra lateral (menu) */
[data-testid="stSidebar"] {{
    background-color: rgba(26, 26, 26, 0.3); /* Barra lateral escura e ligeiramente transparente (0.7 de opacidade) */
}}

/* Estilo para o botão Realizar Predição */
.stButton > button {{
    background-color: rgba(26, 26, 26, 0.7); /* Fundo do botão escuro, quase preto, ligeiramente transparente */
    color: #ADD8E6; /* Cor do texto do botão: Azul claro para contraste */
    border: 2px solid #00008B; /* Borda do botão: Azul escuro (Navy Blue) */
    border-radius: 8px; /* Cantos arredondados */
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease; /* Transição suave para todas as propriedades */
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3); /* Sombra */
}}

/* Estilo para o botão ao passar o mouse */
.stButton > button:hover {{
    border-color: #1E90FF; /* Borda do botão ao passar o mouse: Azul */
    color: white; /* Cor do texto ao passar o mouse: Branco */
}}

</style>
""", unsafe_allow_html=True)



st.title("⚽ Resultados do Brasileirão Série A") # Título principal atualizado
st.markdown("Este aplicativo permite prever o resultado de partidas de futebol com base em um modelo de Machine Learning treinado com dados combinados.")
st.markdown("Por favor, **escolha um Time Mandante e um Time Visitante (obrigatório)**. As outras opções são opcionais para refinar a predição.")

# Carrega modelo salvo
@st.cache_resource
def load_trained_model():
    """Carrega o modelo de Machine Learning treinado."""
    try:
        model_path = os.path.join("model/brasileirao_classificacao.pkl")
        if not os.path.exists(f"{model_path}.pkl"):
            st.error(f"Erro: O arquivo do modelo '{model_path}.pkl' não foi encontrado. Certifique-se de que ele está no caminho correto.")
            st.stop()
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de ML: {e}. Verifique a integridade do arquivo 'modelo-final.pkl' e a compatibilidade das versões das bibliotecas.")
        st.stop()

# Carrega dataset limpo
@st.cache_data
def load_dataset():
    """Carrega e pré-processa o dataset tratado."""
    try:
        csv_path = os.path.join("data/BRA_brasileirao_final.csv") # Nome do arquivo salvo no treinamento
        if not os.path.exists(csv_path):
            st.error(f"Erro: O arquivo '{csv_path}' não foi encontrado. Certifique-se de que ele está no caminho correto.")
            st.stop()
        df = pd.read_csv(csv_path)

        # --- Garante a coluna 'competicao' ---
        # Se o CSV é apenas do Brasileirão, podemos criar a coluna aqui.
        if 'competicao' not in df.columns:
            df['competicao'] = 'Brasileirao' # Valor padrão se a coluna não existir
            # Removido o st.warning conforme solicitado
        
        # Garante que 'ano' e 'mes' sejam inteiros
        if 'ano' in df.columns:
            df['ano'] = df['ano'].astype(int)
        if 'mes' in df.columns:
            df['mes'] = df['mes'].astype(int)
        
        # --- Tratamento de colunas numéricas e categóricas (para garantir consistência) ---
        # Lista de todas as colunas que você forneceu, categorizadas
        numeric_cols = [
            'publico', 'publico_max', 'colocacao_mandante', 'colocacao_visitante',
            'valor_equipe_titular_mandante', 'valor_equipe_titular_visitante',
            'idade_media_titular_mandante', 'idade_media_titular_visitante',
            'gols_1_tempo_mandante', 'gols_1_tempo_visitante',
            'escanteios_mandante', 'escanteios_visitante', 'faltas_mandante', 'faltas_visitante',
            'chutes_bola_parada_mandante', 'chutes_bola_parada_visitante', 'defesas_mandante', 'defesas_visitante',
            'impedimentos_mandante', 'impedimentos_visitante', 'chutes_mandante', 'chutes_visitante',
            'chutes_fora_mandante', 'chutes_fora_visitante', 'rodada', 'ano', 'mes'
        ]
        categorical_cols = [
            'estadio', 'arbitro', 'time_mandante', 'time_visitante',
            'tecnico_mandante', 'tecnico_visitante', 'competicao' # 'competicao' agora é tratada aqui
        ]
        
        # Converte colunas numéricas e preenche NaNs com a média
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    df[col].fillna(df[col].mean(), inplace=True)
            # else: print(f"AVISO: Coluna numérica '{col}' não encontrada no CSV.") # Para depuração
        
        # Converte colunas categóricas e preenche NaNs com a moda
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(r'^\s*$', np.nan, regex=True) # Trata strings vazias
                if df[col].isnull().any():
                    if not df[col].mode().empty:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                        df[col].fillna('Desconhecido', inplace=True) # Fallback se a moda for vazia
            # else: print(f"AVISO: Coluna categórica '{col}' não encontrada no CSV.") # Para depuração

        return df
    except Exception as e:
        st.error(f"Erro ao carregar ou pré-processar o dataset: {e}")
        st.stop()

df = load_dataset()
model = load_trained_model()

# Inferência de valores padrão (moda/média) para colunas que não são input do usuário
def get_default_inputs(df_data, model_features):
    defaults = {}
    for col in model_features:
        if col in df_data.columns:
            if pd.api.types.is_numeric_dtype(df_data[col]):
                defaults[col] = df_data[col].mean()
            else:
                defaults[col] = df_data[col].mode()[0] if not df_data[col].mode().empty else "Desconhecido"
        else:
            # Fallback para colunas que o modelo espera mas não estão no df_data
            # ou foram ignoradas no setup do PyCaret mas são esperadas no input do predict_model.
            # Tente inferir o tipo para dar um valor padrão adequado.
            if any(k in col for k in ["idade", "valor", "publico", "colocacao", "gols", "escanteios", "faltas", "chutes", "defesas", "impedimentos", "rodada", "ano", "mes"]):
                defaults[col] = 0.0 # Valor numérico padrão
            else:
                defaults[col] = "Desconhecido" # Valor categórico padrão
    return defaults

# Tenta obter os nomes das features de forma mais robusta
# O PyCaret retorna um pipeline, e as features de entrada são geralmente acessíveis
# através do passo final do pré-processamento ou do próprio modelo se for um estimador sklearn.
# A forma mais segura é usar o predict_model com um DataFrame vazio para obter as colunas esperadas.
try:
    # Cria um DataFrame vazio com as colunas que o modelo espera
    # Isso é um truque para obter os nomes das features que o pipeline do PyCaret espera.
    # predict_model com data=None ou um DataFrame vazio pode causar erro se o pipeline não for robusto.
    # A melhor forma é inspecionar o pipeline do PyCaret.
    # No PyCaret, após o setup, exp.get_config('X_train').columns contém as features originais.
    # Mas para o modelo carregado, podemos tentar:
    
    # Opção 1: Se o modelo for um pipeline Scikit-learn (o que o PyCaret geralmente salva)
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
    elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_names_in_'):
        # Se for um pipeline e o último passo tiver feature_names_in_
        model_features = model.steps[-1][1].feature_names_in_
    else:
        # Fallback: Criar um DataFrame de exemplo e ver quais colunas o predict_model espera
        # Isso pode ser complexo, então vamos tentar com um conjunto conhecido de colunas
        # que o seu setup do PyCaret ignora ou usa.
        
        # Lista de todas as features que o seu modelo *deve* ter visto no setup,
        # excluindo as ignoradas e a target.
        # Esta lista deve ser mantida consistente com o script de treinamento.
        # Colunas que o modelo espera:
        expected_features = [
            'publico', 'publico_max', 'colocacao_mandante', 'colocacao_visitante',
            'valor_equipe_titular_mandante', 'valor_equipe_titular_visitante',
            'idade_media_titular_mandante', 'idade_media_titular_visitante',
            'gols_1_tempo_mandante', 'gols_1_tempo_visitante',
            'escanteios_mandante', 'escanteios_visitante', 'faltas_mandante', 'faltas_visitante',
            'chutes_bola_parada_mandante', 'chutes_bola_parada_visitante', 'defesas_mandante', 'defesas_visitante',
            'impedimentos_mandante', 'impedimentos_visitante', 'chutes_mandante', 'chutes_visitante',
            'chutes_fora_mandante', 'chutes_fora_visitante',
            'rodada', 'ano', 'mes',
            'estadio', 'arbitro', 'time_mandante', 'time_visitante',
            'tecnico_mandante', 'tecnico_visitante', 'competicao'
            # Note: gols_mandante, gols_visitante, resultado, resultado_ajustado, data são ignoradas ou target
        ]
        model_features = [f for f in expected_features if f in df.columns] # Garante que as features existem no df carregado
        
        # Se o modelo foi treinado com um subconjunto específico, esta lista precisaria ser mais precisa.
        # Uma forma mais robusta seria salvar a lista de features junto com o modelo.
        # Por agora, esta lista é uma estimativa baseada no seu script de treinamento.
        st.warning("Não foi possível aceder a 'feature_names_in_'. Usando uma lista predefinida de features. Verifique a consistência.")

    defaults = get_default_inputs(df, model_features)

except Exception as e:
    st.error(f"Erro ao determinar as features de entrada do modelo: {e}")
    st.warning("Isso pode ser devido a uma incompatibilidade de versão entre o PyCaret/Scikit-learn ou um problema no modelo salvo.")
    st.stop()


# --- Sidebar para input do usuário ---
st.sidebar.header("Parâmetros da Partida")

# Campeonato (fixo, não é um selectbox)
competicao_fixa = "Brasileirao" 

# Filtrar todos os times do DataFrame completo, já que a competição é fixa
all_teams_in_competition = sorted(pd.concat([df['time_mandante'], df['time_visitante']]).dropna().unique())

if not all_teams_in_competition:
    st.error(f"Não há times disponíveis no dataset. Verifique o arquivo CSV.")
    st.stop()

time_mandante = st.sidebar.selectbox("Time Mandante", all_teams_in_competition)
time_visitante = st.sidebar.selectbox("Time Visitante", [t for t in all_teams_in_competition if t != time_mandante])

# Ano da Partida
min_year_data = int(df["ano"].min()) if not df["ano"].empty else 2003 # Ano mínimo do seu dataset
min_year_display = 2003 # Ano mínimo para exibir no seletor, conforme solicitado
max_year_prediction = 2050 # Definido para 2050

# Garante que a lista de anos começa em 2003 e vai até 2050
list_of_years = list(range(min_year_display, max_year_prediction + 1))
# Define o índice padrão para 2003 (que é o primeiro elemento da lista)
ano = st.sidebar.selectbox("Ano", list_of_years, index=list_of_years.index(2003) if 2003 in list_of_years else 0) # Padrão para 2003
st.sidebar.text(f"Mínimo: {min_year_display} | Máximo: {max_year_prediction}") # Mensagem para mostrar o ano máximo

# Mês da Partida
meses_disponiveis = sorted(df["mes"].unique())
mes = st.sidebar.selectbox("Mês", meses_disponiveis)

# Rodada
min_rodada = int(df["rodada"].min()) if "rodada" in df.columns and not df["rodada"].empty else 1
max_rodada = int(df["rodada"].max()) if "rodada" in df.columns and not df["rodada"].empty else 38
rodada = st.sidebar.number_input("Rodada", min_value=min_rodada, max_value=max_rodada, value=min_rodada)


btn = st.button("🔮 Realizar Predição")

if btn:
    # Cria o DataFrame de input para a predição, usando os defaults e sobrescrevendo com os inputs do usuário
    input_data = pd.DataFrame([defaults])
    
    input_data["time_mandante"] = time_mandante
    input_data["time_visitante"] = time_visitante
    input_data["ano"] = ano
    input_data["mes"] = mes
    input_data["rodada"] = rodada
    input_data["competicao"] = competicao_fixa # Usa a competição fixa "Brasileirao"

    try:
        resultado = predict_model(model, data=input_data)
        pred_label = resultado["prediction_label"].iloc[0]
        prob = resultado["prediction_score"].iloc[0]

        # Lógica de desempate baseada no histórico de vitórias (mantida do seu código)
        if pred_label == "Empate":
            # Certifica-se de que 'resultado' está disponível e que os times existem
            vitorias_mandante_hist = df[df["time_mandante"] == time_mandante]["resultado"].value_counts().get("Casa", 0)
            vitorias_visitante_hist = df[df["time_visitante"] == time_visitante]["resultado"].value_counts().get("Fora", 0)

            if vitorias_mandante_hist > vitorias_visitante_hist:
                pred_label = "Casa"
            elif vitorias_visitante_hist > vitorias_mandante_hist:
                pred_label = "Fora"
            else:
                pred_label = "Empate" # Mantém empate se o histórico de vitórias for igual

        st.subheader("Resultado Previsto:")
        if pred_label == "Casa":
            st.success(f"🏠 Vitória do {time_mandante}")
        elif pred_label == "Fora":
            st.warning(f"🚗 Vitória do {time_visitante}")
        else:
            st.info("🤝 Empate")
        st.caption(f"Probabilidade do modelo: {prob * 100:.2f}% (antes do ajuste histórico)")

        with st.expander("🔎 Dados de entrada utilizados"):
            st.dataframe(input_data.T, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao prever: {e}")
        st.write("Detalhes do erro:", e)
        st.write("Dados de entrada para predição:", input_data)
