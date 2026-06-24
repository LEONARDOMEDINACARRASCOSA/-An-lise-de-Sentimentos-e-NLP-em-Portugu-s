import streamlit as st
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator

# =====================================================
# DOWNLOAD DOS RECURSOS NLTK
# =====================================================
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)

# =====================================================
# CONFIGURAÇÃO DA PÁGINA (Deve ser o primeiro comando Streamlit)
# =====================================================
st.set_page_config(
    page_title="Análise de Sentimentos",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# CARREGAMENTO DO MODELO SPACY
# =====================================================
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    st.error(
        "Modelo 'pt_core_news_sm' não encontrado.\n\n"
        "Execute no terminal:\n"
        "python -m spacy download pt_core_news_sm"
    )
    st.stop()

# =====================================================
# INTERFACE PRINCIPAL
# =====================================================
st.title("🧠 Análise de Sentimentos e NLP em Português")
st.markdown(
    "Digite um texto em português para realizar análise de sentimentos "
    "e processamento de linguagem natural."
)

# ENTRADA DO USUÁRIO
texto_pt = st.text_area(
    "Digite seu texto:",
    value="Eu adoro este produto! É maravilhoso.",
    height=150
)

# =====================================================
# PROCESSAMENTO E EXIBIÇÃO
# =====================================================
if st.button("🚀 Analisar Texto"):

    if not texto_pt.strip():
        st.warning("Digite algum texto para análise.")
        st.stop()

    with st.spinner("Processando texto..."):
        # -------------------------------------------------
        # NLP COM SPACY
        # -------------------------------------------------
        documento = nlp(texto_pt)

        # Tokenização usando SpaCy
        tokens = [token.text for token in documento]

        # Stopwords
        stops = set(stopwords.words("portuguese"))

        palavras_uteis = [
            token.text
            for token in documento
            if token.is_alpha and token.text.lower() not in stops
        ]

        # -------------------------------------------------
        # TRADUÇÃO + VADER
        # -------------------------------------------------
        try:
            texto_en = GoogleTranslator(
                source="pt",
                target="en"
            ).translate(texto_pt)

            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(texto_en)
            score_compound = scores["compound"]

            if score_compound >= 0.05:
                resultado = "🟢 Positivo"
            elif score_compound <= -0.05:
                resultado = "🔴 Negativo"
            else:
                resultado = "🟡 Neutro"

        except Exception as e:
            st.error(f"Erro durante a tradução/análise: {e}")
            st.stop()

    # =====================================================
    # RESULTADOS (Agora dentro do bloco do botão)
    # =====================================================
    st.markdown("---")
    st.subheader("📈 Resultado da Análise")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Sentimento", value=resultado)
    with col2:
        st.metric(label="Score Compound", value=f"{score_compound:.4f}")

    st.info(f"**Texto traduzido para análise:** {texto_en}")

    # =====================================================
    # ABAS
    # =====================================================
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs([
        "📝 Tokens",
        "🔍 Classes Gramaticais",
        "📊 Scores Completos"
    ])

    # -------------------------------------------------
    # ABA 1
    # -------------------------------------------------
    with tab1:
        st.subheader("Tokens encontrados")
        st.write(tokens)
        st.subheader("Palavras úteis (sem stopwords)")
        st.write(palavras_uteis)

    # -------------------------------------------------
    # ABA 2
    # -------------------------------------------------
    with tab2:
        st.subheader("Análise Gramatical (POS Tags)")
        for token in documento:
            explicacao = spacy.explain(token.pos_) if spacy.explain(token.pos_) else "Sem descrição"
            st.write(f"• **{token.text}** → {token.pos_} ({explicacao})")

    # -------------------------------------------------
    # ABA 3
    # -------------------------------------------------
    with tab3:
        st.subheader("Scores completos do VADER")
        st.json(scores)