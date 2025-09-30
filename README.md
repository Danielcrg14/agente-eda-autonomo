# 🤖 Agente EDA Autônomo

Sistema inteligente para Análise Exploratória de Dados com capacidades avançadas de Q&A específico.

## 🌐 ACESSO DIRETO (ONLINE)

**Link público funcionando**: https://agente-eda-autonomo-dsqspejfiearctjcsc8ef.streamlit.app

👆 **Clique no link acima para usar imediatamente!**

## 🚀 Como Usar Online

1. **Acesse** o link público acima
2. **Faça upload** de qualquer arquivo CSV
3. **Aguarde** a análise automática
4. **Faça perguntas** específicas sobre seus dados
5. **Visualize** gráficos gerados automaticamente

## 🔧 Instalação Local (Opcional)

Se quiser executar localmente:

### Pré-requisitos
- Python 3.11+
- Chave OpenAI válida

### Instalação
```bash
# 1. Clonar repositório
git clone https://github.com/Danielcrsg14/agente-eda-autonomo.git
cd agente-eda-autonomo

# 2. Criar ambiente virtual
python -m venv venv

# 3. Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Instalar dependências
pip install -r requirements.txt

# 5. Configurar chave OpenAI
# Criar arquivo .env com:
# OPENAI_API_KEY=sua_chave_openai_aqui

# 6. Executar dashboard
streamlit run dashboard.py
Acesso Local
URL: http://localhost:8501
🧠 Capacidades do Agente
8 Ferramentas Especializadas
carregar_csv - Carregamento inteligente + detecção automática
analisar_automaticamente - EDA completa contextualizada
criar_grafico_automatico - Visualizações adaptativas
obter_contexto_atual - Contexto da tabela + memória
analisar_variavel_especifica - Análise granular de colunas
analisar_tendencias_temporais - Séries temporais + sazonalidade
detectar_clusters - K-means automático
resposta_direta - Q&A específico
Tipos de Dados Suportados
🚨 Fraude/Segurança - Desbalanceamento, outliers
🏪 Vendas/Comercial - Performance, produtos
🔬 Científico - Classificação, correlações
👥 RH - Demographics, salários
🏥 Médico - Estatísticas, correlações
📅 Temporal - Tendências, sazonalidade
🎯 Geral - Estatísticas descritivas
💬 Exemplos de Perguntas
Perguntas Específicas Suportadas
"Sobre o que é esta tabela?"
"Qual a média da coluna X?"
"Quais são os outliers da coluna Y?"
"Detecte agrupamentos nos dados"
"Analise especificamente a variável Z"
"Identifique correlações entre variáveis"
"Crie gráficos apropriados para meus dados"
🎯 Demonstração
Teste com Diferentes Tipos de CSV
Upload qualquer CSV (vendas, médico, financeiro, etc.)
Veja detecção automática do tipo
Explore análises contextualizadas
Faça perguntas específicas
Visualize gráficos adaptativos
📊 Resultados Comprovados
Generalização Testada
✅ Fraude (284k linhas) - Análise de segurança
✅ Vendas (9k linhas) - Análise comercial
✅ Médico (1k linhas) - Análise científica
Capacidades Avançadas
Outliers detectados: 31,904 (método IQR)
Clusters identificados: 5 grupos (K-means)
Correlações calculadas: Matriz completa
Memória funcional: Contexto entre perguntas
🏆 Tecnologias
Framework: LangChain + AgentExecutor
LLM: OpenAI GPT-4o-mini
Interface: Streamlit
Análise: Pandas + NumPy + Scikit-learn
Visualização: Matplotlib + Seaborn
📞 Contato
Desenvolvido para o Institut d'Intelligence Artificielle Appliquée

Atividade: Agentes Autônomos - EDA
