# dashboard.py CORRIGIDO COM KEYS - PARTE 1: CABEÇALHO
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import base64
import importlib
import glob

# CORREÇÃO: Import dinâmico para evitar cache
from agente_eda import perguntar_ao_agente

# Configuração da página
st.set_page_config(
    page_title="🤖 Agente EDA - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .stats-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho principal
st.markdown("""
<div class="main-header">
    <h1>🤖 Agente EDA Autônomo</h1>
    <p style="color: white; margin: 0;">Análise Exploratória de Dados com Inteligência Artificial</p>
</div>
""", unsafe_allow_html=True)
# dashboard.py CORRIGIDO COM KEYS - PARTE 2: FUNÇÕES COM KEYS ÚNICAS

def resetar_agente():
    """Reseta o agente para limpar cache entre análises"""
    try:
        import agente_eda
        
        # Limpar variáveis globais
        agente_eda.dataset_atual = None
        agente_eda.descobertas_memoria = []
        
        # CORREÇÃO: Limpar APENAS grafico_atual.png
        if os.path.exists('grafico_atual.png'):
            try:
                os.remove('grafico_atual.png')
                st.info("🗑️ Gráfico anterior removido")
            except:
                pass
        
        # Forçar reload do módulo
        importlib.reload(agente_eda)
        
        from agente_eda import perguntar_ao_agente
        
        return perguntar_ao_agente, True
        
    except Exception as e:
        st.warning(f"⚠️ Aviso ao resetar agente: {str(e)}")
        from agente_eda import perguntar_ao_agente
        return perguntar_ao_agente, False

def exibir_arquivo_info(uploaded_file):
    """Exibe informações sobre o arquivo carregado"""
    st.markdown(f"""
    <div class="success-box">
        <h4>📄 Arquivo Carregado:</h4>
        <ul>
            <li><strong>Nome:</strong> {uploaded_file.name}</li>
            <li><strong>Tamanho:</strong> {uploaded_file.size / 1024:.1f} KB</li>
            <li><strong>Tipo:</strong> {uploaded_file.type}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def criar_sidebar():
    """Cria a barra lateral com upload e instruções"""
    st.sidebar.markdown("# 📁 Carregar Dados")
    
    uploaded_file = st.sidebar.file_uploader(
        "📎 Escolha um arquivo CSV",
        type=['csv'],
        help="Faça upload de qualquer arquivo CSV para análise automática"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Como funciona:")
    
    steps = [
        "📤 **Faça upload** do seu CSV",
        "⚡ **Aguarde** a análise automática",
        "📊 **Veja** os insights e gráficos",
        "💬 **Faça perguntas** personalizadas"
    ]
    
    for i, step in enumerate(steps, 1):
        st.sidebar.markdown(f"{i}. {step}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Capacidades do Agente:")
    st.sidebar.markdown("- 🔍 **Detecção automática** de 6+ tipos de dados")
    st.sidebar.markdown("- 📊 **Estatísticas descritivas** completas")
    st.sidebar.markdown("- 🎯 **Clustering** e padrões")
    st.sidebar.markdown("- 🚨 **Detecção de outliers**")
    st.sidebar.markdown("- 📈 **Análise temporal**")
    st.sidebar.markdown("- 💬 **Q&A específico**")
    
    return uploaded_file

def mostrar_metricas_dataset(resposta_agente):
    """Extrai e exibe métricas dinâmicas do dataset"""
    st.markdown("### 📊 Métricas do Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Extrair informações da resposta
    linhas = "Analisando..."
    colunas = "Verificando..."
    tipo = "Detectando..."
    status = "Processando..."
    
    if resposta_agente:
        # Extrair dados da resposta
        if "Linhas:" in resposta_agente:
            try:
                linhas_texto = resposta_agente.split("Linhas:")[1].split("\n")[0].strip()
                linhas = linhas_texto.split()[0].replace(",", "")
            except:
                pass
        
        if "Colunas:" in resposta_agente:
            try:
                colunas_texto = resposta_agente.split("Colunas:")[1].split("\n")[0].strip()
                colunas = colunas_texto.split()[0]
            except:
                pass
        
        if "Tipo detectado:" in resposta_agente:
            try:
                tipo = resposta_agente.split("Tipo detectado:")[1].split("\n")[0].strip()
                if len(tipo) > 25:
                    tipo = tipo[:22] + "..."
            except:
                pass
        
        status = "Concluído ✅"
    
    with col1:
        st.metric(label="📋 Total de Linhas", value=linhas, delta="Dataset carregado")
    
    with col2:
        st.metric(label="📊 Colunas", value=colunas, delta="Estruturadas")
    
    with col3:
        st.metric(label="🔍 Tipo Detectado", value=tipo, delta="Auto-detecção")
    
    with col4:
        st.metric(label="📈 Status", value=status, delta="Análise completa")

def exibir_analise_principal(resposta):
    """Exibe a análise principal do agente"""
    st.markdown("## 🧠 Análise Automática do Agente")
    
    tab1, tab2, tab3 = st.tabs(["📋 Resumo Executivo", "📊 Análise Detalhada", "🎯 Insights"])
    
    with tab1:
        st.markdown("### 🎯 Resumo Executivo")
        if resposta and "Tipo detectado:" in resposta:
            tipo = resposta.split("Tipo detectado:")[1].split("\n")[0].strip()
            st.success(f"🎯 **Tipo Detectado**: {tipo}")
        
        if resposta:
            resumo = '\n'.join(resposta.split('\n')[:10])
            st.markdown(f"```\n{resumo}\n```")
    
    with tab2:
        st.markdown("### 📊 Análise Detalhada")
        if resposta:
            st.markdown(resposta)
    
    with tab3:
        st.markdown("### 🎯 Principais Insights")
        if resposta and "INSIGHTS AUTOMÁTICOS:" in resposta:
            insights = resposta.split("INSIGHTS AUTOMÁTICOS:")[1]
            st.markdown(insights)

def exibir_graficos():
    """Exibe APENAS o gráfico atual (SEM BOTÕES DUPLICADOS)"""
    st.markdown("## 📈 Visualizações Automáticas")
    
    if os.path.exists('grafico_atual.png'):
        st.success("✅ Gráfico criado pelo agente!")
        st.image('grafico_atual.png', use_container_width=True)
        
        with st.expander("ℹ️ Sobre este gráfico"):
            st.markdown("""
            **Gráfico automático baseado no tipo de dados:**
            - 🏪 **Vendas**: Histogramas + ranking produtos
            - 🚨 **Fraude**: Distribuição classes + valores
            - 🔬 **Científico**: Scatter plots + classificações
            - 📊 **Geral**: Correlações + distribuições
            
            O agente escolhe automaticamente a visualização mais apropriada.
            """)
    else:
        st.info("📊 Aguardando gráfico do agente...")
        st.markdown("""
        **O agente criará visualizações automaticamente:**
        1. 🔍 Detecta tipo de dados
        2. 📊 Escolhe gráfico apropriado  
        3. 🎨 Cria visualização específica
        """)

def secao_perguntas():
    """Seção para perguntas personalizadas"""
    st.markdown("---")
    st.markdown("## 💬 Perguntas Personalizadas")
    
    exemplos_perguntas = [
        "Sobre o que é esta tabela?",
        "Qual a média da coluna Sales?",
        "Quais são os outliers da coluna Amount?",
        "Detecte agrupamentos nos dados",
        "Crie um gráfico de correlação"
    ]
    
    cols = st.columns(2)
    for i, pergunta in enumerate(exemplos_perguntas):
        with cols[i % 2]:
            if st.button(f"📝 {pergunta}", key=f"btn_exemplo_{i}"):
                st.session_state.pergunta_selecionada = pergunta
    
    pergunta_default = st.session_state.get('pergunta_selecionada', '')
    pergunta_usuario = st.text_area(
        "🤔 Sua pergunta:",
        value=pergunta_default,
        placeholder="Ex: Qual a distribuição da coluna X?",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🚀 Perguntar", type="primary", key="btn_perguntar_principal"):
            if pergunta_usuario.strip():
                return pergunta_usuario
            else:
                st.warning("⚠️ Digite uma pergunta primeiro!")
    
    with col2:
        if st.button("🗑️ Limpar", key="btn_limpar_pergunta"):
            st.session_state.pergunta_selecionada = ''
            st.rerun()
    
    return None

def pagina_inicial():
    """Página inicial"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">👋 Bem-vindo ao Agente EDA Autônomo!</h2>
        <p style="color: white; margin: 0.5rem 0 0 0;">
            Análise Exploratória de Dados com Inteligência Artificial
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧠 Inteligência
        - **8 ferramentas** especializadas
        - **Detecção automática** de tipos
        - **Q&A específico**
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Análises
        - **Estatísticas** completas
        - **Outliers** (IQR)
        - **Clustering** (K-means)
        """)
    
    with col3:
        st.markdown("""
        ### 💬 Perguntas
        - **"Sobre a tabela?"**
        - **"Média da coluna X?"**
        - **"Outliers de Y?"**
        """)
    
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Demonstração com dataset de fraude:**
        - 📊 284.807 transações
        - 🎯 Todas as capacidades
        """)
    
    with col2:
        if st.button("🧪 Demonstrar", type="primary", key="btn_demo_inicial"):
            return "demo"
    
    return None

def executar_demonstracao():
    """Demonstração com dataset de exemplo"""
    st.markdown("## 🧪 Demonstração")
    
    with st.spinner("🔄 Resetando..."):
        perguntar_ao_agente_func, reset_ok = resetar_agente()
    
    if reset_ok:
        st.success("✅ Agente resetado!")
    
    with st.spinner('🧠 Analisando exemplo...'):
        resposta = perguntar_ao_agente_func("Carregue o arquivo data/creditcard.csv e faça análise completa com gráficos")
    
    mostrar_metricas_dataset(resposta)
    exibir_analise_principal(resposta)
    exibir_graficos()
    
    st.markdown("---")
    st.markdown("## 💬 Teste Perguntas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("❓ Sobre esta tabela?", key="btn_demo_contexto"):
            with st.spinner('🤔 Respondendo...'):
                resposta_contexto = perguntar_ao_agente_func("Sobre o que é esta tabela?")
            st.markdown("### 🤖 Resposta:")
            st.markdown(resposta_contexto)
    
    with col2:
        if st.button("❓ Média da coluna Amount?", key="btn_demo_media"):
            with st.spinner('🤔 Calculando...'):
                resposta_media = perguntar_ao_agente_func("Qual a média da coluna Amount?")
            st.markdown("### 🤖 Resposta:")
            st.markdown(resposta_media)
    
    pergunta = secao_perguntas()
    
    if pergunta:
        st.markdown("### 🤖 Resposta:")
        with st.spinner('🤔 Pensando...'):
            resposta_personalizada = perguntar_ao_agente_func(pergunta)
        st.markdown(resposta_personalizada)
        exibir_graficos()
        # dashboard.py CORRIGIDO COM KEYS - PARTE 4: MAIN FUNCTION

def main():
    """Função principal do dashboard"""
    
    # Inicializar session state
    if 'pergunta_selecionada' not in st.session_state:
        st.session_state.pergunta_selecionada = ''
    
    # Criar sidebar e obter arquivo
    uploaded_file = criar_sidebar()
    
    # LÓGICA PRINCIPAL
    if uploaded_file is not None:
        # Arquivo foi carregado
        exibir_arquivo_info(uploaded_file)
        
        st.info("🔄 Preparando agente para análise do novo arquivo...")
        
        # Reset do agente
        perguntar_ao_agente_func, reset_success = resetar_agente()
        
        if reset_success:
            st.success("✅ Agente resetado com sucesso!")
        else:
            st.warning("⚠️ Reset parcial - continuando análise...")
        
        # Salvar arquivo temporariamente
        temp_path = f"temp_{uploaded_file.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Análise automática
            st.markdown("## 🚀 Iniciando Análise Automática")
            with st.spinner('🧠 Agente analisando seus dados...'):
                pergunta = f"Carregue o arquivo {temp_path} e faça uma análise completa com gráficos automáticos"
                resposta = perguntar_ao_agente_func(pergunta)
            
            # Exibir resultados
            mostrar_metricas_dataset(resposta)
            exibir_analise_principal(resposta)
            exibir_graficos()
            
            # Seção de perguntas personalizadas
            st.markdown("---")
            st.markdown("## 💬 Faça Perguntas Específicas")
            
            # Perguntas rápidas com KEYS ÚNICAS
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("❓ Sobre esta tabela", key="btn_main_contexto"):
                    with st.spinner('🤔 Obtendo contexto...'):
                        resposta_contexto = perguntar_ao_agente_func("Sobre o que é esta tabela?")
                    st.markdown("### 🤖 Contexto da Tabela:")
                    st.markdown(resposta_contexto)
            
            with col2:
                if st.button("📊 Detectar Clusters", key="btn_main_clusters"):
                    with st.spinner('🎯 Detectando agrupamentos...'):
                        resposta_clusters = perguntar_ao_agente_func("Detecte agrupamentos nos dados")
                    st.markdown("### 🤖 Análise de Clusters:")
                    st.markdown(resposta_clusters)
                    # Atualizar gráficos após clusters
                    exibir_graficos()
            
            with col3:
                if st.button("🚨 Analisar Outliers", key="btn_main_outliers"):
                    with st.spinner('🔍 Analisando outliers...'):
                        resposta_outliers = perguntar_ao_agente_func("Identifique outliers nas principais colunas numéricas")
                    st.markdown("### 🤖 Análise de Outliers:")
                    st.markdown(resposta_outliers)
            
            # Campo de pergunta livre
            pergunta_usuario = secao_perguntas()
            
            if pergunta_usuario:
                st.markdown("### 🤖 Resposta do Agente:")
                with st.spinner('🤔 Agente pensando...'):
                    resposta_personalizada = perguntar_ao_agente_func(pergunta_usuario)
                
                st.markdown(resposta_personalizada)
                exibir_graficos()
        
        finally:
            # Limpar arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    else:
        # Nenhum arquivo carregado - mostrar página inicial
        acao = pagina_inicial()
        
        if acao == "demo":
            executar_demonstracao()

# ===== EXECUÇÃO PRINCIPAL =====

if __name__ == "__main__":
    main()

# Rodapé final
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 <strong>Agente EDA Autônomo</strong> - Desenvolvido com ❤️ usando Streamlit e LangChain</p>
    <p>📊 Análise Exploratória de Dados com Inteligência Artificial</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem;">
        🎯 <em>Sistema com gráfico único por análise - sem mistura de contextos</em>
    </p>
    <p style="font-size: 0.8rem;">
        💬 <em>Faça perguntas específicas sobre qualquer aspecto dos dados</em>
    </p>
</div>
""", unsafe_allow_html=True)