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
