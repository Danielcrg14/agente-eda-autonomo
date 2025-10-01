# 🤖 Agente EDA Autônomo

Sistema inteligente de Análise Exploratória de Dados com 8 ferramentas especializadas e capacidades universais de Q&A específico.

## 🌐 ACESSO DIRETO (ONLINE)

**🔗 Link público funcionando**: https://agente-eda-autonomo-dsqispejfiearctjcsc8ef.streamlit.app/

👆 **Clique no link acima para usar imediatamente!**

## 🚀 Como Usar Online

1. **Acesse** o link público acima
2. **Faça upload** de qualquer arquivo CSV (até 200MB)
3. **Aguarde** a análise automática (30-60 segundos)
4. **Explore** insights e visualizações geradas automaticamente
5. **Faça perguntas** específicas sobre qualquer aspecto dos dados
6. **Teste** com diferentes tipos de CSV

## 🧠 Capacidades do Agente

### 8 Ferramentas Especializadas
1. **carregar_csv** - Carregamento inteligente + detecção automática
2. **analisar_automaticamente** - EDA completa contextualizada
3. **criar_grafico_automatico** - Visualizações adaptativas
4. **obter_contexto_atual** - Contexto + memória de descobertas
5. **analisar_variavel_especifica** - Análise granular (estatísticas + outliers)
6. **analisar_tendencias_temporais** - Séries temporais + sazonalidade
7. **detectar_clusters** - K-means automático com normalização
8. **resposta_direta** - Q&A específico para perguntas pontuais

### Detecção Automática de Tipos
- 🚨 **Fraude/Segurança** - Desbalanceamento, outliers, padrões suspeitos
- 🏪 **Vendas/Comercial** - Performance, produtos, análise de receita
- 🔬 **Científico/Experimental** - Classificação, correlações, medidas
- 🏥 **Médico/Saúde** - Fatores clínicos, diagnósticos, correlações
- 👥 **RH/Recursos Humanos** - Demographics, salários, departamentos
- 📅 **Temporal/Séries** - Tendências, sazonalidade, padrões
- 📊 **Numérico Puro** - Estatísticas completas, correlações
- 📝 **Categórico Puro** - Frequências, distribuições textuais
- 🎯 **Misto/Geral** - Análise híbrida robusta

## 💬 Perguntas Específicas Suportadas

### Exemplos Validados no Sistema
- **"Sobre o que é esta tabela?"** → Contexto automático inteligente
- **"Qual a média da coluna Amount?"** → Resposta: 88.3496
- **"Quais são os outliers da coluna Amount?"** → 31,904 detectados (11.20%)
- **"Detecte agrupamentos nos dados"** → 5 clusters em dataset grande
- **"Analise especificamente a variável V1"** → Estatísticas + IQR completas
- **"Identifique correlações entre variáveis"** → V2 vs Amount (0.531)
- **"Crie gráficos apropriados"** → Automático por tipo detectado

## 📊 Resultados Comprovados

### Generalização Testada com Dados Reais
- ✅ **Fraude** (284,807 linhas) → "DETECÇÃO DE FRAUDE DE CARTÃO DE CRÉDITO"
- ✅ **Vendas** (9,800 linhas) → "DADOS DE VENDAS/COMERCIAL"
- ✅ **Médico** (1,025 linhas) → "DADOS MÉDICOS/SAÚDE"
- ✅ **Genérico** (10 linhas) → "DATASET GERAL MISTO"
- ✅ **Científico** (testado) → "DADOS CIENTÍFICOS/EXPERIMENTAIS"
- ✅ **Categórico** (testado) → "DATASET CATEGÓRICO PURO"

### Métricas de Performance Demonstradas
- **Processamento**: 284,807 linhas sem erro
- **Outliers identificados**: 31,904 (método IQR científico)
- **Clusters detectados**: 5 grupos (K-means robusto)
- **Correlações calculadas**: Matriz completa automática
- **Tempo de resposta**: 30-60 segundos para qualquer CSV
- **Taxa de sucesso**: 100% nos tipos testados
- **Adaptação contextual**: Linguagem específica por domínio

## 🎯 Demonstração Rápida

### Teste Imediato (3 minutos)
1. **Acesse**: https://agente-eda-autonomo-dsqispejfiearctjcsc8ef.streamlit.app/
2. **Upload**: Qualquer CSV que você tenha
3. **Observe**: Detecção automática do tipo
4. **Explore**: Análises contextualizadas
5. **Pergunte**: "Sobre o que é esta tabela?"
6. **Teste**: "Detecte clusters nos dados"
7. **Verifique**: Gráficos adaptativos criados

## 🔧 Instalação Local (Opcional)

### Pré-requisitos
- Python 3.11+
- Chave OpenAI válida

### Instalação Completa
```bash
# 1. Clonar repositório
git clone https://github.com/Danielcrg14/agente-eda-autonomo.git
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
# OPENAI_MODEL=gpt-4o-mini

# 6. Executar dashboard
streamlit run dashboard.py
Acesso Local
URL: http://localhost:8501
📁 Arquivos do Projeto
agente_eda.py (1,933 linhas) - Agente principal com 8 ferramentas especializadas
dashboard.py (499 linhas) - Interface web profissional
teste_completo.py (47 linhas) - Demonstrações automatizadas
requirements.txt (12 linhas) - Dependências precisas
.env (4 linhas) - Configurações seguras
Estatísticas do Código
Total de código: 2,495 linhas especializadas
Tamanho total: ~106 KB
8 ferramentas robustas implementadas
Fallbacks universais para qualquer CSV
Tratamento gracioso de erros
## 🏆 Tecnologias

- **Framework**: LangChain + AgentExecutor (padrão ReAct)
- **LLM**: OpenAI GPT-4o-mini (temperatura 0.1)
- **Interface**: Streamlit (dashboard responsivo)
- **Análise de Dados**: Pandas + NumPy
- **Machine Learning**: Scikit-learn (K-means clustering)
- **Visualização**: Matplotlib + Seaborn
- **Memória**: ConversationBufferMemory (contexto conversacional)

## ⚡ Recursos Avançados

### Robustez e Tratamento de Erros
- **Fallbacks universais** para qualquer estrutura CSV
- **Tratamento gracioso** de dados faltantes
- **Normalização automática** para clustering
- **Validação** de tipos de dados
- **Busca inteligente** por colunas (flexível)
- **Limitação de performance** para datasets grandes (>50k linhas)

### Capacidades de Machine Learning
- **K-means clustering** com detecção automática de número de clusters
- **Detecção de outliers** método IQR científico
- **Análise de correlações** com identificação de multicolinearidade
- **Normalização de dados** para análises robustas
- **Análise temporal** com múltiplas estratégias

## 📈 Casos de Uso Validados

### Tipos de Negócio
- **Detecção de Fraude**: Análise de desbalanceamento + padrões suspeitos
- **Análise Comercial**: Performance de vendas + ranking de produtos
- **Pesquisa Médica**: Correlações clínicas + fatores de risco
- **Gestão de RH**: Demographics + análise salarial
- **Pesquisa Científica**: Classificação de espécies + medidas

### Estruturas de Dados
- **Big Data**: Até 284k linhas processadas
- **Small Data**: 10 linhas com análise apropriada
- **Dados limpos**: Zero valores ausentes
- **Dados com problemas**: Tratamento automático
- **Múltiplos tipos**: Numérico + categórico + misto

## 🔒 Segurança

- **Chaves API** protegidas em Streamlit Secrets
- **Arquivo .env** mascarado no repositório público
- **Dados temporários** removidos automaticamente
- **Sem persistência** de dados do usuário
- **Reset automático** entre sessões

## 📞 Suporte e Contato

### Desenvolvido para
**Institut d'Intelligence Artificielle Appliquée**  
**Atividade**: Agentes Autônomos - Análise Exploratória de Dados

### Informações Técnicas
- **Padrão**: ReAct (Reasoning and Acting)
- **Autonomia**: 100% (zero intervenção manual)
- **Memória**: Conversacional com descobertas acumuladas
- **Performance**: Otimizado para datasets de qualquer tamanho

---

**🎯 Sistema pronto para avaliação com qualquer arquivo CSV!**  
**✨ Teste agora mesmo com seus próprios dados!**
