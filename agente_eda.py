# agente_eda.py CORRIGIDO - PARTE 1: IMPORTS E CONFIGURAÇÃO
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Imports do LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

# Carregar configurações
load_dotenv()

print("🚀 Iniciando Agente EDA...")

# Variáveis globais simples
dataset_atual = None
descobertas_memoria = []

# Configurar LLM
llm = ChatOpenAI(
    model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
    temperature=0.1,
    api_key=os.getenv('OPENAI_API_KEY')
)

print(f"🤖 LLM configurado: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
print("✅ Agente básico inicializado!")

# ===== FERRAMENTAS INTELIGENTES =====
# agente_eda.py CORRIGIDO - PARTE 2: FERRAMENTA CARREGAR CSV

@tool
def carregar_csv(caminho_arquivo: str) -> str:
    """
    🔧 Carrega um arquivo CSV e faz análise inicial automática.
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo CSV (ex: 'data/creditcard.csv')
    
    Returns:
        str: Relatório da análise inicial
    """
    global dataset_atual, descobertas_memoria
    
    try:
        print(f"📊 Carregando: {caminho_arquivo}")
        
        # Limpar gráficos antigos ANTES de carregar novo dataset
        import glob
        graficos_antigos = glob.glob("grafico_*.png")
        for grafico in graficos_antigos:
            try:
                os.remove(grafico)
            except:
                pass
        
        # Carregar o dataset
        df = pd.read_csv(caminho_arquivo)
        dataset_atual = df
        
        # Análise inicial automática
        relatorio = f"""
🎯 DATASET CARREGADO COM SUCESSO!

📋 INFORMAÇÕES BÁSICAS:
- Arquivo: {caminho_arquivo}
- Linhas: {df.shape[0]:,}
- Colunas: {df.shape[1]}
- Tamanho em memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

📊 ESTRUTURA DOS DADOS:
- Colunas numéricas: {len(df.select_dtypes(include=[np.number]).columns)}
- Colunas de texto: {len(df.select_dtypes(include=['object']).columns)}
- Valores ausentes total: {df.isnull().sum().sum()}

🔍 NOMES DAS COLUNAS:
{', '.join(df.columns.tolist())}

🧠 DETECÇÃO AUTOMÁTICA DE TIPO:"""

        # Auto-detecção inteligente MELHORADA do tipo de dataset
        colunas_lower = [col.lower() for col in df.columns]
        colunas_texto = ' '.join(colunas_lower)
        
        # Detecção de FRAUDE
        if 'class' in colunas_lower and any('v' in col.lower() for col in df.columns):
            tipo_detectado = "DETECÇÃO DE FRAUDE DE CARTÃO DE CRÉDITO"
            relatorio += f"""
- Tipo detectado: {tipo_detectado}
- Análises recomendadas: distribuição de fraudes, padrões nas variáveis V, análise de valores
- Foco especial: desbalanceamento de classes, outliers, correlações
"""
        
        # Detecção de VENDAS/COMERCIAL
        elif any(palavra in colunas_texto for palavra in ['sales', 'price', 'revenue', 'product', 'quantity', 'customer']):
            tipo_detectado = "DADOS DE VENDAS/COMERCIAL"
            relatorio += f"""
- Tipo detectado: {tipo_detectado}
- Análises recomendadas: tendências de vendas, análise por produto, performance comercial
- Foco especial: sazonalidade, ranking de produtos, análise de receita
"""
        
        # Detecção de RH/RECURSOS HUMANOS
        elif any(palavra in colunas_texto for palavra in ['salary', 'employee', 'department', 'age', 'years']):
            tipo_detectado = "DADOS DE RH/RECURSOS HUMANOS"
            relatorio += f"""
- Tipo detectado: {tipo_detectado}
- Análises recomendadas: análise salarial, distribuição por departamento, demographics
- Foco especial: equidade salarial, performance por área, análise de idade
"""
        
        # Detecção de DADOS CIENTÍFICOS
        elif any(palavra in colunas_texto for palavra in ['species', 'petal', 'sepal', 'length', 'width', 'class']):
            tipo_detectado = "DADOS CIENTÍFICOS/EXPERIMENTAIS"
            relatorio += f"""
- Tipo detectado: {tipo_detectado}
- Análises recomendadas: distribuições por classe, correlações entre medidas
- Foco especial: classificação de espécies, análise morfométrica, clusters
"""
        
        # Detecção de DADOS TEMPORAIS
        elif any(palavra in colunas_texto for palavra in ['date', 'time', 'timestamp', 'year', 'month']):
            tipo_detectado = "DADOS TEMPORAIS/SÉRIES TEMPORAIS"
            relatorio += f"""
- Tipo detectado: {tipo_detectado}
- Análises recomendadas: tendências temporais, sazonalidade, previsões
- Foco especial: análise de séries, detecção de padrões, decomposição temporal
"""
        
        # Fallback para DADOS GERAIS
        else:
            tipo_detectado = "DATASET GERAL"
            relatorio += f"""
- Tipo detectado: {tipo_detectado}
- Análises recomendadas: estatísticas descritivas, correlações, distribuições
- Foco especial: análise exploratória abrangente, identificação de padrões
"""
        
        # Salvar descoberta na memória
        descoberta = f"Dataset carregado: {caminho_arquivo} ({df.shape[0]} linhas, tipo: {tipo_detectado})"
        descobertas_memoria.append(descoberta)
        
        return relatorio
        
    except Exception as e:
        return f"❌ ERRO ao carregar {caminho_arquivo}: {str(e)}"

print("🔧 Ferramenta 'carregar_csv' criada!")
# agente_eda.py CORRIGIDO - PARTE 3: FERRAMENTA ANÁLISE AUTOMÁTICA

@tool
def analisar_automaticamente() -> str:
    """
    🧠 Faz análise automática completa do dataset carregado.
    O agente decide sozinho quais análises fazer.
    
    Returns:
        str: Relatório completo da análise automática
    """
    global dataset_atual, descobertas_memoria
    
    if dataset_atual is None:
        return "❌ Nenhum dataset carregado! Use 'carregar_csv' primeiro."
    
    df = dataset_atual
    relatorio = "🧠 ANÁLISE AUTOMÁTICA INTELIGENTE\n" + "="*50 + "\n"
    
    try:
        # 1. Estatísticas básicas
        relatorio += "\n📊 1. ESTATÍSTICAS DESCRITIVAS:\n"
        colunas_numericas = df.select_dtypes(include=[np.number]).columns
        
        if len(colunas_numericas) > 0:
            desc = df[colunas_numericas].describe()
            relatorio += f"Colunas numéricas analisadas: {len(colunas_numericas)}\n"
            relatorio += desc.to_string()
        
        # 2. Análise específica por tipo de dados
        colunas_lower = [col.lower() for col in df.columns]
        colunas_texto = ' '.join(colunas_lower)
        
        # Análise para FRAUDE
        if 'class' in colunas_lower and any('v' in col.lower() for col in df.columns):
            relatorio += "\n\n🎯 2. ANÁLISE DE FRAUDE DETECTADA:\n"
            if 'Class' in df.columns:
                contagem = df['Class'].value_counts()
                total = len(df)
                
                relatorio += f"- Transações normais (0): {contagem[0]:,} ({contagem[0]/total*100:.2f}%)\n"
                relatorio += f"- Transações fraudulentas (1): {contagem[1]:,} ({contagem[1]/total*100:.2f}%)\n"
                relatorio += f"- Taxa de fraude: {contagem[1]/total*100:.4f}%\n"
                
                if contagem[1]/total < 0.01:
                    relatorio += "⚠️  DATASET ALTAMENTE DESBALANCEADO - Fraudes são muito raras!\n"
        
        # Análise para VENDAS
        elif any(palavra in colunas_texto for palavra in ['sales', 'price', 'revenue', 'product', 'quantity']):
            relatorio += "\n\n🏪 2. ANÁLISE DE VENDAS DETECTADA:\n"
            
            # Procurar colunas de vendas
            colunas_vendas = [col for col in df.columns if any(palavra in col.lower() for palavra in ['sales', 'revenue', 'price', 'amount'])]
            if colunas_vendas:
                col_vendas = colunas_vendas[0]
                vendas = df[col_vendas]
                relatorio += f"- Coluna de vendas identificada: {col_vendas}\n"
                relatorio += f"- Vendas totais: ${vendas.sum():,.2f}\n"
                relatorio += f"- Vendas médias: ${vendas.mean():.2f}\n"
                relatorio += f"- Maior venda: ${vendas.max():.2f}\n"
                relatorio += f"- Menor venda: ${vendas.min():.2f}\n"
            
            # Procurar produtos
            colunas_produto = [col for col in df.columns if any(palavra in col.lower() for palavra in ['product', 'item', 'categoria'])]
            if colunas_produto:
                col_produto = colunas_produto[0]
                produtos_unicos = df[col_produto].nunique()
                relatorio += f"- Produtos únicos: {produtos_unicos}\n"
                top_produtos = df[col_produto].value_counts().head(3)
                relatorio += f"- Top 3 produtos mais frequentes: {', '.join(top_produtos.index.tolist())}\n"
        
        # Análise para DADOS CIENTÍFICOS
        elif any(palavra in colunas_texto for palavra in ['species', 'petal', 'sepal', 'length', 'width']):
            relatorio += "\n\n🔬 2. ANÁLISE CIENTÍFICA DETECTADA:\n"
            
            # Procurar coluna de classes/espécies
            colunas_classe = [col for col in df.columns if any(palavra in col.lower() for palavra in ['species', 'class', 'tipo'])]
            if colunas_classe:
                col_classe = colunas_classe[0]
                classes = df[col_classe].value_counts()
                relatorio += f"- Classes/Espécies identificadas: {classes.index.tolist()}\n"
                relatorio += f"- Distribuição por classe:\n"
                for classe, count in classes.items():
                    relatorio += f"  * {classe}: {count} ({count/len(df)*100:.1f}%)\n"
            
            # Análise de medidas morfométricas
            colunas_medidas = [col for col in df.columns if any(palavra in col.lower() for palavra in ['length', 'width', 'height', 'petal', 'sepal'])]
            if len(colunas_medidas) >= 2:
                relatorio += f"- Medidas morfométricas encontradas: {', '.join(colunas_medidas)}\n"
                correlacao_max = df[colunas_medidas].corr().abs().max().max()
                relatorio += f"- Correlação máxima entre medidas: {correlacao_max:.3f}\n"
        
        # 3. Análise de valores ausentes
        relatorio += "\n\n🔍 3. VALORES AUSENTES:\n"
        valores_ausentes = df.isnull().sum()
        if valores_ausentes.sum() == 0:
            relatorio += "✅ Nenhum valor ausente encontrado!\n"
        else:
            relatorio += "⚠️  Valores ausentes encontrados:\n"
            for col, missing in valores_ausentes[valores_ausentes > 0].items():
                relatorio += f"   - {col}: {missing} ({missing/len(df)*100:.2f}%)\n"
        
        # 4. Análise de correlações (para dados numéricos)
        if len(colunas_numericas) > 1:
            relatorio += "\n\n🔗 4. ANÁLISE DE CORRELAÇÕES:\n"
            corr_matrix = df[colunas_numericas].corr()
            
            # Encontrar correlações mais fortes (excluindo diagonal)
            corr_values = corr_matrix.abs().values
            np.fill_diagonal(corr_values, 0)
            max_corr = np.max(corr_values)
            max_idx = np.unravel_index(np.argmax(corr_values), corr_values.shape)
            
            col1 = colunas_numericas[max_idx[0]]
            col2 = colunas_numericas[max_idx[1]]
            
            relatorio += f"- Correlação mais forte: {col1} vs {col2} ({max_corr:.3f})\n"
            
            if max_corr > 0.7:
                relatorio += "⚠️  Correlação muito alta detectada - possível multicolinearidade\n"
            elif max_corr > 0.5:
                relatorio += "💡 Correlação moderada detectada - variáveis relacionadas\n"
            else:
                relatorio += "✅ Correlações baixas - variáveis independentes\n"
        
        # 5. Insights automáticos MELHORADOS
        relatorio += "\n\n🧠 5. INSIGHTS AUTOMÁTICOS:\n"
        insights = []
        
        # Insights para FRAUDE
        if 'Class' in df.columns:
            fraud_rate = df['Class'].sum() / len(df)
            if fraud_rate < 0.001:
                insights.append("- Dataset extremamente desbalanceado - técnicas especiais podem ser necessárias")
            
            if 'Amount' in df.columns:
                normal_avg = df[df['Class'] == 0]['Amount'].mean()
                fraud_avg = df[df['Class'] == 1]['Amount'].mean()
                if fraud_avg < normal_avg:
                    insights.append("- Transações fraudulentas tendem a ter valores MENORES que as normais")
                else:
                    insights.append("- Transações fraudulentas tendem a ter valores MAIORES que as normais")
        
        # Insights para VENDAS
        elif any(palavra in colunas_texto for palavra in ['sales', 'price', 'revenue']):
            colunas_vendas = [col for col in df.columns if any(palavra in col.lower() for palavra in ['sales', 'revenue', 'price'])]
            if colunas_vendas:
                col_vendas = colunas_vendas[0]
                cv = df[col_vendas].std() / df[col_vendas].mean()
                if cv > 1:
                    insights.append("- Alta variabilidade nas vendas - mercado instável ou sazonalidade")
                else:
                    insights.append("- Vendas com variabilidade moderada - padrão consistente")
        
        # Insights para DADOS CIENTÍFICOS
        elif any(palavra in colunas_texto for palavra in ['species', 'petal', 'sepal']):
            insights.append("- Dataset científico identificado - ideal para análise de classificação")
            if len(colunas_numericas) >= 4:
                insights.append("- Múltiplas medidas disponíveis - possível análise multivariada")
        
        # Insights GERAIS
        if len(colunas_numericas) > len(df.select_dtypes(include=['object']).columns):
            insights.append("- Dataset predominantemente numérico - ideal para análises estatísticas")
        
        # Verificar variáveis PCA (V1, V2, etc.)
        v_columns = [col for col in df.columns if col.startswith('V')]
        if len(v_columns) > 10:
            insights.append(f"- Dataset contém {len(v_columns)} variáveis transformadas por PCA (V1-V{len(v_columns)})")
            insights.append("- Essas variáveis são resultado de transformação para proteger dados sensíveis")
        
        # Insights sobre tamanho do dataset
        if len(df) > 100000:
            insights.append("- Dataset grande (>100k linhas) - análises robustas possíveis")
        elif len(df) < 1000:
            insights.append("- Dataset pequeno (<1k linhas) - cuidado com generalizações")
        
        for insight in insights:
            relatorio += f"{insight}\n"
        
        # Salvar na memória
        descoberta = f"Análise automática realizada: {len(insights)} insights gerados"
        descobertas_memoria.append(descoberta)
        
        return relatorio
        
    except Exception as e:
        return f"❌ ERRO na análise automática: {str(e)}"

print("🧠 Ferramenta 'analisar_automaticamente' criada!")
# agente_eda.py CORRIGIDO - PARTE 4: FERRAMENTA GRÁFICOS (ARQUIVO ÚNICO)

@tool
def criar_grafico_automatico(tipo_analise: str = "auto") -> str:
    """
    📊 Cria gráficos automáticos baseados no tipo de análise solicitada.
    
    Args:
        tipo_analise (str): Tipo de gráfico - "auto", "distribuicao", "correlacao", "valores"
    
    Returns:
        str: Relatório sobre o gráfico criado
    """
    global dataset_atual, descobertas_memoria
    
    if dataset_atual is None:
        return "❌ Nenhum dataset carregado! Use 'carregar_csv' primeiro."
    
    df = dataset_atual
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('default')
        colunas_lower = [col.lower() for col in df.columns]
        colunas_texto = ' '.join(colunas_lower)
        
        # CORREÇÃO: Auto-detecção do tipo de gráfico apropriado
        if tipo_analise == "auto":
            # Para dados de FRAUDE
            if 'class' in colunas_lower and any('v' in col.lower() for col in df.columns) and 'Class' in df.columns:
                tipo_analise = "distribuicao_fraude"
            
            # Para dados de VENDAS
            elif any(palavra in colunas_texto for palavra in ['sales', 'price', 'revenue', 'product', 'quantity']):
                tipo_analise = "vendas_analise"
            
            # Para dados CIENTÍFICOS
            elif any(palavra in colunas_texto for palavra in ['species', 'petal', 'sepal']):
                tipo_analise = "cientifico_analise"
            
            # Para dados GERAIS - usar correlação
            else:
                tipo_analise = "correlacao"
        
        # GRÁFICO PARA FRAUDE
        if tipo_analise == "distribuicao_fraude" and 'Class' in df.columns:
            plt.figure(figsize=(12, 5))
            
            # Subplot 1: Pizza
            plt.subplot(1, 3, 1)
            contagem = df['Class'].value_counts()
            colors = ['lightblue', 'red']
            plt.pie(contagem.values, labels=['Normal (0)', 'Fraude (1)'], 
                   autopct='%1.2f%%', colors=colors, startangle=90)
            plt.title('Distribuição de Classes')
            
            # Subplot 2: Barras
            plt.subplot(1, 3, 2)
            plt.bar(['Normal', 'Fraude'], contagem.values, color=colors)
            plt.title('Contagem por Tipo')
            plt.ylabel('Quantidade')
            
            # Subplot 3: Valores (se Amount existe)
            if 'Amount' in df.columns:
                plt.subplot(1, 3, 3)
                normal_amount = df[df['Class'] == 0]['Amount']
                fraud_amount = df[df['Class'] == 1]['Amount']
                
                plt.boxplot([normal_amount, fraud_amount], labels=['Normal', 'Fraude'])
                plt.title('Distribuição de Valores')
                plt.ylabel('Valor ($)')
                plt.yscale('log')
            
            plt.tight_layout()
            # CORREÇÃO: Nome único do arquivo
            plt.savefig('grafico_atual.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            descoberta = "Gráfico de fraude criado: grafico_atual.png"
            descobertas_memoria.append(descoberta)
            
            return "📊 GRÁFICO DE FRAUDE CRIADO! Arquivo: grafico_atual.png - Análise completa de distribuição e valores."
        
        # GRÁFICO PARA VENDAS (CORRIGIDO)
        elif tipo_analise == "vendas_analise":
            plt.figure(figsize=(15, 5))
            
            # Encontrar colunas relevantes para vendas
            colunas_vendas = [col for col in df.columns if any(palavra in col.lower() for palavra in ['sales', 'revenue', 'price', 'amount'])]
            colunas_produto = [col for col in df.columns if any(palavra in col.lower() for palavra in ['product', 'item', 'categoria', 'category'])]
            
            if colunas_vendas:
                col_vendas = colunas_vendas[0]
                
                # Subplot 1: Histograma de vendas
                plt.subplot(1, 3, 1)
                plt.hist(df[col_vendas], bins=30, color='green', alpha=0.7)
                plt.title(f'Distribuição de {col_vendas}')
                plt.xlabel(col_vendas)
                plt.ylabel('Frequência')
                
                # Subplot 2: Top produtos (se disponível)
                if colunas_produto:
                    plt.subplot(1, 3, 2)
                    col_produto = colunas_produto[0]
                    top_produtos = df[col_produto].value_counts().head(10)
                    y_pos = range(len(top_produtos))
                    plt.barh(y_pos, top_produtos.values, color='orange')
                    plt.yticks(y_pos, [str(x)[:20] + '...' if len(str(x)) > 20 else str(x) for x in top_produtos.index])
                    plt.title(f'Top 10 {col_produto}')
                    plt.xlabel('Quantidade')
                
                # Subplot 3: Box plot de vendas
                plt.subplot(1, 3, 3)
                plt.boxplot(df[col_vendas])
                plt.title(f'Box Plot - {col_vendas}')
                plt.ylabel(col_vendas)
            
            plt.tight_layout()
            # CORREÇÃO: Nome único do arquivo
            plt.savefig('grafico_atual.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            descoberta = "Gráfico de vendas criado: grafico_atual.png"
            descobertas_memoria.append(descoberta)
            
            return "📊 GRÁFICO DE VENDAS CRIADO! Arquivo: grafico_atual.png - Análise completa de vendas e produtos."
        
        # GRÁFICO PARA DADOS CIENTÍFICOS
        elif tipo_analise == "cientifico_analise":
            plt.figure(figsize=(15, 5))
            
            colunas_medidas = [col for col in df.columns if any(palavra in col.lower() for palavra in ['length', 'width', 'petal', 'sepal'])]
            colunas_classe = [col for col in df.columns if any(palavra in col.lower() for palavra in ['species', 'class'])]
            
            if len(colunas_medidas) >= 2:
                # Subplot 1: Scatter plot
                plt.subplot(1, 3, 1)
                if colunas_classe:
                    classes = df[colunas_classe[0]].unique()
                    colors = ['red', 'blue', 'green', 'orange', 'purple']
                    for i, classe in enumerate(classes):
                        mask = df[colunas_classe[0]] == classe
                        plt.scatter(df[mask][colunas_medidas[0]], df[mask][colunas_medidas[1]], 
                                  c=colors[i % len(colors)], label=classe, alpha=0.7)
                    plt.legend()
                else:
                    plt.scatter(df[colunas_medidas[0]], df[colunas_medidas[1]], alpha=0.7)
                
                plt.xlabel(colunas_medidas[0])
                plt.ylabel(colunas_medidas[1])
                plt.title('Scatter Plot - Medidas')
                
                # Subplot 2: Histograma das medidas
                plt.subplot(1, 3, 2)
                for i, col in enumerate(colunas_medidas[:4]):
                    plt.hist(df[col], alpha=0.5, label=col, bins=20)
                plt.legend()
                plt.title('Distribuições das Medidas')
                plt.xlabel('Valores')
                plt.ylabel('Frequência')
                
                # Subplot 3: Box plot por classe
                if colunas_classe and len(colunas_medidas) >= 1:
                    plt.subplot(1, 3, 3)
                    classes = df[colunas_classe[0]].unique()
                    data_boxplot = []
                    labels_boxplot = []
                    
                    for classe in classes:
                        mask = df[colunas_classe[0]] == classe
                        data_boxplot.append(df[mask][colunas_medidas[0]])
                        labels_boxplot.append(classe)
                    
                    plt.boxplot(data_boxplot, labels=labels_boxplot)
                    plt.title(f'{colunas_medidas[0]} por {colunas_classe[0]}')
                    plt.ylabel(colunas_medidas[0])
            
            plt.tight_layout()
            # CORREÇÃO: Nome único do arquivo
            plt.savefig('grafico_atual.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            descoberta = "Gráfico científico criado: grafico_atual.png"
            descobertas_memoria.append(descoberta)
            
            return "📊 GRÁFICO CIENTÍFICO CRIADO! Arquivo: grafico_atual.png - Análise de medidas e classificações."
        
        # GRÁFICO DE CORRELAÇÃO (GERAL) - CORRIGIDO
        elif tipo_analise == "correlacao":
            colunas_numericas = df.select_dtypes(include=[np.number]).columns
            
            if len(colunas_numericas) > 1:
                plt.figure(figsize=(12, 8))
                
                # Calcular matriz de correlação
                corr_matrix = df[colunas_numericas].corr()
                
                # Criar heatmap
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, fmt='.2f')
                
                plt.title('Matriz de Correlação')
                plt.tight_layout()
                # CORREÇÃO: Nome único do arquivo
                plt.savefig('grafico_atual.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                descoberta = "Gráfico de correlação criado: grafico_atual.png"
                descobertas_memoria.append(descoberta)
                
                return "📊 GRÁFICO DE CORRELAÇÃO CRIADO! Arquivo: grafico_atual.png - Matriz de correlação entre variáveis numéricas."
            
            else:
                return "❌ Dados insuficientes para criar matriz de correlação (precisa de 2+ colunas numéricas)."
        
        # FALLBACK - Gráfico simples de distribuições (CORRIGIDO)
        else:
            colunas_numericas = df.select_dtypes(include=[np.number]).columns
            
            if len(colunas_numericas) > 0:
                plt.figure(figsize=(12, 8))
                
                # Histogramas das primeiras 6 colunas numéricas
                n_cols = min(6, len(colunas_numericas))
                rows = 2
                cols = 3
                
                for i, col in enumerate(colunas_numericas[:n_cols]):
                    plt.subplot(rows, cols, i+1)
                    plt.hist(df[col], bins=20, alpha=0.7, color=f'C{i}')
                    plt.title(f'Distribuição - {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequência')
                
                plt.tight_layout()
                # CORREÇÃO: Nome único do arquivo
                plt.savefig('grafico_atual.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                descoberta = "Gráfico geral criado: grafico_atual.png"
                descobertas_memoria.append(descoberta)
                
                return "📊 GRÁFICO GERAL CRIADO! Arquivo: grafico_atual.png - Distribuições das principais variáveis."
            
            else:
                return "❌ Nenhuma coluna numérica encontrada para criar gráficos."
            
    except Exception as e:
        return f"❌ ERRO ao criar gráfico: {str(e)}"

print("📊 Ferramenta 'criar_grafico_automatico' criada!")
# agente_eda.py CORRIGIDO - PARTE 5: FERRAMENTAS AUXILIARES

@tool
def analisar_variavel_especifica(nome_variavel: str, tipo_analise: str = "completa") -> str:
    """
    🔍 Analisa uma variável específica do dataset carregado.
    
    Args:
        nome_variavel (str): Nome exato da coluna a ser analisada
        tipo_analise (str): "completa", "distribuicao", "outliers", "estatisticas"
    
    Returns:
        str: Análise detalhada da variável
    """
    global dataset_atual
    
    if dataset_atual is None:
        return "❌ Nenhum dataset carregado! Use 'carregar_csv' primeiro."
    
    df = dataset_atual
    
    # Verificar se a variável existe
    if nome_variavel not in df.columns:
        colunas_similares = [col for col in df.columns if nome_variavel.lower() in col.lower()]
        if colunas_similares:
            return f"❌ Variável '{nome_variavel}' não encontrada. Você quis dizer: {', '.join(colunas_similares)}?"
        else:
            return f"❌ Variável '{nome_variavel}' não existe. Colunas disponíveis: {', '.join(df.columns.tolist())}"
    
    var = df[nome_variavel]
    
    relatorio = f"🔍 ANÁLISE DA VARIÁVEL: {nome_variavel}\n" + "="*40 + "\n"
    
    # Informações básicas
    relatorio += f"\n📊 INFORMAÇÕES BÁSICAS:\n"
    relatorio += f"- Tipo de dados: {var.dtype}\n"
    relatorio += f"- Valores únicos: {var.nunique():,}\n"
    relatorio += f"- Valores não-nulos: {var.count():,}\n"
    relatorio += f"- Valores ausentes: {var.isnull().sum()}\n"
    
    if tipo_analise in ["completa", "estatisticas"]:
        # Estatísticas para variáveis numéricas
        if var.dtype in ['int64', 'float64']:
            relatorio += f"\n📈 ESTATÍSTICAS DESCRITIVAS:\n"
            relatorio += f"- Média: {var.mean():.4f}\n"
            relatorio += f"- Mediana: {var.median():.4f}\n"
            relatorio += f"- Mínimo: {var.min():.4f}\n"
            relatorio += f"- Máximo: {var.max():.4f}\n"
            relatorio += f"- Desvio padrão: {var.std():.4f}\n"
            relatorio += f"- Variância: {var.var():.4f}\n"
            relatorio += f"- Assimetria: {var.skew():.4f}\n"
            relatorio += f"- Curtose: {var.kurtosis():.4f}\n"
            
            # Quartis
            q1 = var.quantile(0.25)
            q3 = var.quantile(0.75)
            iqr = q3 - q1
            relatorio += f"- Q1 (25%): {q1:.4f}\n"
            relatorio += f"- Q3 (75%): {q3:.4f}\n"
            relatorio += f"- IQR: {iqr:.4f}\n"
        
        # Estatísticas para variáveis categóricas
        else:
            relatorio += f"\n📝 ANÁLISE CATEGÓRICA:\n"
            value_counts = var.value_counts().head(10)
            relatorio += f"- Top 10 valores mais frequentes:\n"
            for valor, freq in value_counts.items():
                relatorio += f"  * {valor}: {freq} ({freq/len(var)*100:.2f}%)\n"
    
    if tipo_analise in ["completa", "outliers"]:
        # Detecção de outliers (apenas para variáveis numéricas)
        if var.dtype in ['int64', 'float64']:
            relatorio += f"\n🚨 DETECÇÃO DE OUTLIERS:\n"
            
            # Método IQR
            q1 = var.quantile(0.25)
            q3 = var.quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr
            
            outliers = var[(var < limite_inferior) | (var > limite_superior)]
            relatorio += f"- Outliers detectados (IQR): {len(outliers)}\n"
            
            if len(outliers) > 0:
                relatorio += f"- Percentual de outliers: {len(outliers)/len(var)*100:.2f}%\n"
                relatorio += f"- Limite inferior: {limite_inferior:.4f}\n"
                relatorio += f"- Limite superior: {limite_superior:.4f}\n"
                
                if len(outliers) <= 10:
                    relatorio += f"- Valores outliers: {outliers.tolist()}\n"
                else:
                    relatorio += f"- Primeiros 5 outliers: {outliers.head().tolist()}\n"
    
    return relatorio

print("🔍 Ferramenta 'analisar_variavel_especifica' criada!")

@tool
def obter_contexto_atual() -> str:
    """
    🧠 Obtém informações sobre o dataset atualmente carregado.
    
    Returns:
        str: Contexto atual do dataset
    """
    global dataset_atual, descobertas_memoria
    
    if dataset_atual is None:
        return "❌ Nenhum dataset carregado no momento."
    
    df = dataset_atual
    
    # Detectar tipo
    colunas_lower = [col.lower() for col in df.columns]
    colunas_texto = ' '.join(colunas_lower)
    
    if any(palavra in colunas_texto for palavra in ['sales', 'price', 'revenue', 'product']):
        contexto = "Este é um dataset de VENDAS COMERCIAIS com informações sobre produtos, clientes e transações de vendas."
    elif 'class' in colunas_lower and any('v' in col.lower() for col in df.columns):
        contexto = "Este é um dataset de DETECÇÃO DE FRAUDE de cartão de crédito."
    elif any(palavra in colunas_texto for palavra in ['species', 'petal', 'sepal']):
        contexto = "Este é um dataset CIENTÍFICO com dados de classificação e medidas."
    else:
        contexto = "Este é um dataset GERAL para análise exploratória."
    
    relatorio = f"""
🔍 SOBRE ESTA TABELA:

📊 CONTEXTO:
{contexto}

📋 CARACTERÍSTICAS:
- Linhas: {df.shape[0]:,}
- Colunas: {df.shape[1]}
- Colunas: {', '.join(df.columns.tolist())}
"""
    
    return relatorio

print("🧠 Ferramenta 'obter_contexto_atual' criada!")
# agente_eda.py CORRIGIDO - PARTE 6: FERRAMENTAS AVANÇADAS

@tool
def analisar_tendencias_temporais(coluna_data: str = "auto", coluna_valor: str = "auto") -> str:
    """
    📅 Analisa tendências temporais nos dados.
    
    Args:
        coluna_data (str): Nome da coluna de data/tempo ("auto" para detecção automática)
        coluna_valor (str): Nome da coluna de valores ("auto" para detecção automática)
    
    Returns:
        str: Análise de tendências temporais
    """
    global dataset_atual, descobertas_memoria
    
    if dataset_atual is None:
        return "❌ Nenhum dataset carregado! Use 'carregar_csv' primeiro."
    
    df = dataset_atual
    
    try:
        # Auto-detecção de colunas temporais
        if coluna_data == "auto":
            colunas_temporais = []
            for col in df.columns:
                col_lower = col.lower()
                if any(palavra in col_lower for palavra in ['date', 'time', 'timestamp', 'year', 'month', 'day']):
                    colunas_temporais.append(col)
            
            if not colunas_temporais:
                return "❌ Nenhuma coluna temporal detectada. Colunas disponíveis: " + ", ".join(df.columns.tolist())
            
            coluna_data = colunas_temporais[0]
        
        # Auto-detecção de coluna de valores
        if coluna_valor == "auto":
            colunas_valor = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64'] and col != coluna_data:
                    col_lower = col.lower()
                    if any(palavra in col_lower for palavra in ['sales', 'amount', 'price', 'revenue', 'value']):
                        colunas_valor.append(col)
            
            if not colunas_valor:
                # Pegar primeira coluna numérica que não é a data
                colunas_numericas = df.select_dtypes(include=[np.number]).columns
                colunas_valor = [col for col in colunas_numericas if col != coluna_data]
            
            if not colunas_valor:
                return "❌ Nenhuma coluna de valores numéricos encontrada para análise temporal."
            
            coluna_valor = colunas_valor[0]
        
        # Verificar se as colunas existem
        if coluna_data not in df.columns:
            return f"❌ Coluna de data '{coluna_data}' não encontrada."
        if coluna_valor not in df.columns:
            return f"❌ Coluna de valores '{coluna_valor}' não encontrada."
        
        relatorio = f"📅 ANÁLISE TEMPORAL\n" + "="*40 + "\n"
        relatorio += f"\n🔍 COLUNAS ANALISADAS:\n"
        relatorio += f"- Coluna temporal: {coluna_data}\n"
        relatorio += f"- Coluna de valores: {coluna_valor}\n"
        
        # Tentar converter para datetime
        try:
            df_temp = df.copy()
            df_temp[coluna_data] = pd.to_datetime(df_temp[coluna_data])
            
            # Análise temporal básica
            relatorio += f"\n📊 ANÁLISE TEMPORAL:\n"
            relatorio += f"- Período inicial: {df_temp[coluna_data].min()}\n"
            relatorio += f"- Período final: {df_temp[coluna_data].max()}\n"
            relatorio += f"- Duração total: {(df_temp[coluna_data].max() - df_temp[coluna_data].min()).days} dias\n"
            
            # Criar gráfico temporal
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            
            # Agrupar por mês para visualização
            df_mensal = df_temp.groupby([df_temp[coluna_data].dt.to_period('M')])[coluna_valor].sum()
            
            plt.subplot(1, 2, 1)
            df_mensal.plot(kind='line', color='blue')
            plt.title(f'Tendência Temporal - {coluna_valor}')
            plt.xlabel('Período')
            plt.ylabel(coluna_valor)
            plt.xticks(rotation=45)
            
            # Histograma por mês do ano (sazonalidade)
            plt.subplot(1, 2, 2)
            df_temp['mes'] = df_temp[coluna_data].dt.month
            sazonalidade = df_temp.groupby('mes')[coluna_valor].mean()
            meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                    'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            plt.bar(range(1, 13), [sazonalidade.get(i, 0) for i in range(1, 13)], color='green')
            plt.title('Sazonalidade por Mês')
            plt.xlabel('Mês')
            plt.ylabel(f'Média {coluna_valor}')
            plt.xticks(range(1, 13), meses, rotation=45)
            
            plt.tight_layout()
            # CORREÇÃO: Nome único do arquivo
            plt.savefig('grafico_atual.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            descoberta = f"Análise temporal realizada: {coluna_data} vs {coluna_valor}"
            descobertas_memoria.append(descoberta)
            
            relatorio += f"\n📊 GRÁFICO TEMPORAL CRIADO: grafico_atual.png\n"
            
            return relatorio
            
        except Exception as e_date:
            # Se não conseguir converter para data, análise básica
            relatorio += f"\n⚠️ Não foi possível converter '{coluna_data}' para formato de data.\n"
            relatorio += f"Fazendo análise básica da sequência de valores...\n"
            
            # Análise de tendência simples (assumindo ordem temporal)
            valores = df[coluna_valor]
            if len(valores) > 1:
                primeira_metade = valores[:len(valores)//2].mean()
                segunda_metade = valores[len(valores)//2:].mean()
                mudanca = ((segunda_metade - primeira_metade) / primeira_metade) * 100
                
                relatorio += f"- Valor médio primeira metade: {primeira_metade:.2f}\n"
                relatorio += f"- Valor médio segunda metade: {segunda_metade:.2f}\n"
                relatorio += f"- Mudança percentual: {mudanca:.2f}%\n"
                
                if abs(mudanca) > 10:
                    relatorio += f"⚠️ Tendência significativa detectada!\n"
                else:
                    relatorio += f"✅ Valores relativamente estáveis.\n"
            
            return relatorio
            
    except Exception as e:
        return f"❌ ERRO na análise temporal: {str(e)}"

print("📅 Ferramenta 'analisar_tendencias_temporais' criada!")

@tool
def detectar_clusters(n_clusters: str = "auto", colunas: str = "auto") -> str:
    """
    🎯 Detecta agrupamentos (clusters) nos dados usando K-means.
    
    Args:
        n_clusters (str): Número de clusters ("auto" para detecção automática, ou número específico)
        colunas (str): Colunas para usar ("auto" para seleção automática, ou nomes separados por vírgula)
    
    Returns:
        str: Análise de clusters encontrados
    """
    global dataset_atual, descobertas_memoria
    
    if dataset_atual is None:
        return "❌ Nenhum dataset carregado! Use 'carregar_csv' primeiro."
    
    df = dataset_atual
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Auto-seleção de colunas numéricas
        if colunas == "auto":
            colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remover colunas de ID se existirem
            colunas_para_cluster = [col for col in colunas_numericas 
                                  if not any(palavra in col.lower() for palavra in ['id', 'index', 'row'])]
        else:
            colunas_para_cluster = [col.strip() for col in colunas.split(',')]
        
        if len(colunas_para_cluster) < 2:
            return "❌ Precisa de pelo menos 2 colunas numéricas para análise de clusters."
        
        # Preparar dados (remover NaN)
        dados_cluster = df[colunas_para_cluster].dropna()
        
        if len(dados_cluster) < 10:
            return "❌ Dados insuficientes para análise de clusters (mínimo 10 linhas sem NaN)."
        
        # Normalizar dados
        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(dados_cluster)
        
        relatorio = "🎯 ANÁLISE DE CLUSTERS\n" + "="*40 + "\n"
        relatorio += f"\n🔍 CONFIGURAÇÃO:\n"
        relatorio += f"- Colunas usadas: {', '.join(colunas_para_cluster)}\n"
        relatorio += f"- Linhas analisadas: {len(dados_cluster):,}\n"
        
        # Determinar número de clusters
        if n_clusters == "auto":
            # Método simples para determinar clusters
            best_k = min(5, max(2, len(dados_cluster)//1000))
        else:
            best_k = int(n_clusters)
        
        # Executar clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(dados_normalizados)
        
        relatorio += f"\n📊 RESULTADOS:\n"
        relatorio += f"- Número de clusters encontrados: {best_k}\n"
        relatorio += f"- Distribuição dos clusters:\n"
        
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            relatorio += f"  * Cluster {cluster_id}: {count} pontos ({count/len(clusters)*100:.1f}%)\n"
        
        # Criar gráfico de clusters
        import matplotlib.pyplot as plt
        
        if len(colunas_para_cluster) >= 2:
            plt.figure(figsize=(12, 5))
            
            # Subplot 1: Scatter plot dos clusters
            plt.subplot(1, 2, 1)
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for cluster_id in range(best_k):
                mask = clusters == cluster_id
                plt.scatter(dados_cluster[mask][colunas_para_cluster[0]], 
                          dados_cluster[mask][colunas_para_cluster[1]],
                          c=colors[cluster_id % len(colors)], 
                          label=f'Cluster {cluster_id}', alpha=0.7)
            
            plt.xlabel(colunas_para_cluster[0])
            plt.ylabel(colunas_para_cluster[1])
            plt.title('Clusters Detectados')
            plt.legend()
            
            # Subplot 2: Distribuição dos clusters
            plt.subplot(1, 2, 2)
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            plt.bar(cluster_counts.index, cluster_counts.values, color=colors[:len(cluster_counts)])
            plt.title('Distribuição dos Clusters')
            plt.xlabel('Cluster ID')
            plt.ylabel('Número de Pontos')
            
            plt.tight_layout()
            # CORREÇÃO: Nome único do arquivo
            plt.savefig('grafico_atual.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            relatorio += f"\n📊 GRÁFICO CRIADO: grafico_atual.png\n"
        
        descoberta = f"Análise de clusters realizada: {best_k} clusters detectados"
        descobertas_memoria.append(descoberta)
        
        return relatorio
        
    except ImportError:
        return "❌ Biblioteca scikit-learn não disponível. Instale com: pip install scikit-learn"
    except Exception as e:
        return f"❌ ERRO na análise de clusters: {str(e)}"

print("🎯 Ferramenta 'detectar_clusters' criada!")

@tool
def resposta_direta(pergunta_especifica: str) -> str:
    """
    💬 Responde perguntas específicas e diretas sobre o dataset.
    
    Args:
        pergunta_especifica (str): Pergunta específica sobre estatísticas, valores, etc.
    
    Returns:
        str: Resposta direta e objetiva
    """
    global dataset_atual
    
    if dataset_atual is None:
        return "❌ Nenhum dataset carregado! Use 'carregar_csv' primeiro."
    
    df = dataset_atual
    pergunta_lower = pergunta_especifica.lower()
    
    try:
        # Detectar tipo de pergunta e responder diretamente
        if "sobre o que" in pergunta_lower or "sobre a tabela" in pergunta_lower:
            # Contexto da tabela
            colunas_texto = ' '.join([col.lower() for col in df.columns])
            
            if any(palavra in colunas_texto for palavra in ['sales', 'product', 'revenue']):
                return f"📊 Esta tabela contém DADOS DE VENDAS com {df.shape[0]:,} linhas e {df.shape[1]} colunas. Inclui informações sobre vendas, produtos e transações comerciais."
            elif 'class' in colunas_texto and any('v' in col.lower() for col in df.columns):
                return f"🚨 Esta tabela contém DADOS DE DETECÇÃO DE FRAUDE com {df.shape[0]:,} transações de cartão de crédito para identificar padrões fraudulentos."
            else:
                return f"📈 Esta tabela contém dados para análise exploratória com {df.shape[0]:,} linhas e {df.shape[1]} colunas. Colunas: {', '.join(df.columns.tolist())}"
        
        elif "média" in pergunta_lower:
            # Encontrar coluna mencionada
            for col in df.columns:
                if col.lower() in pergunta_lower:
                    if df[col].dtype in ['int64', 'float64']:
                        return f"📊 A média da coluna '{col}' é: {df[col].mean():.4f}"
                    else:
                        return f"❌ A coluna '{col}' não é numérica (tipo: {df[col].dtype})"
            return "❌ Não consegui identificar qual coluna você quer a média. Especifique o nome da coluna."
        
        elif "máximo" in pergunta_lower or "max" in pergunta_lower:
            for col in df.columns:
                if col.lower() in pergunta_lower:
                    if df[col].dtype in ['int64', 'float64']:
                        return f"📊 O valor máximo da coluna '{col}' é: {df[col].max():.4f}"
                    else:
                        return f"📊 O valor mais frequente da coluna '{col}' é: {df[col].mode().iloc[0]}"
            return "❌ Especifique qual coluna você quer o valor máximo."
        
        elif "mínimo" in pergunta_lower or "min" in pergunta_lower:
            for col in df.columns:
                if col.lower() in pergunta_lower:
                    if df[col].dtype in ['int64', 'float64']:
                        return f"📊 O valor mínimo da coluna '{col}' é: {df[col].min():.4f}"
                    else:
                        return f"📊 O valor menos frequente da coluna '{col}' é: {df[col].value_counts().index[-1]}"
            return "❌ Especifique qual coluna você quer o valor mínimo."
        
        elif "outlier" in pergunta_lower:
            for col in df.columns:
                if col.lower() in pergunta_lower and df[col].dtype in ['int64', 'float64']:
                    # Detectar outliers usando IQR
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    limite_inferior = q1 - 1.5 * iqr
                    limite_superior = q3 + 1.5 * iqr
                    
                    outliers = df[col][(df[col] < limite_inferior) | (df[col] > limite_superior)]
                    
                    return f"🚨 Outliers na coluna '{col}': {len(outliers)} detectados ({len(outliers)/len(df)*100:.2f}% dos dados)"
            
            return "❌ Especifique qual coluna você quer analisar outliers."
        
        else:
            # Resposta genérica
            return f"💡 Para perguntas específicas, tente: 'Qual a média da coluna X?', 'Quais outliers da coluna Y?', 'Sobre o que é a tabela?'"
            
    except Exception as e:
        return f"❌ ERRO ao responder pergunta: {str(e)}"

print("💬 Ferramenta 'resposta_direta' criada!")
# agente_eda.py CORRIGIDO - PARTE 7: CONFIGURAÇÃO DO AGENTE

# ===== CONFIGURAR AGENTE LANGCHAIN =====

def criar_agente_eda():
    """🤖 Cria o agente EDA completo com LangChain"""
    
    # Lista COMPLETA de ferramentas disponíveis
    ferramentas = [
        carregar_csv, 
        analisar_automaticamente, 
        criar_grafico_automatico, 
        obter_contexto_atual, 
        analisar_variavel_especifica, 
        analisar_tendencias_temporais, 
        detectar_clusters, 
        resposta_direta
    ]
    
    # Configurar memória
    memoria = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Prompt FINAL OTIMIZADO para Q&A específico
    prompt = ChatPromptTemplate.from_messages([
        ("system", """🧠 VOCÊ É UM DATA SCIENTIST VIRTUAL ESPECIALIZADO EM EDA (ANÁLISE EXPLORATÓRIA DE DADOS)

SUA MISSÃO:
- Analisar QUALQUER dataset CSV de forma autônoma e inteligente
- Responder perguntas específicas sobre qualquer aspecto dos dados
- Detectar automaticamente o tipo de dados e adaptar suas análises
- Gerar insights valiosos e conclusões próprias
- Criar gráficos automáticos apropriados para cada contexto

SUAS CAPACIDADES (8 FERRAMENTAS):
- carregar_csv: Carrega e faz análise inicial de qualquer CSV
- analisar_automaticamente: Faz análise completa e automática dos dados
- criar_grafico_automatico: Cria gráficos visuais apropriados (use "auto")
- obter_contexto_atual: Obtém informações sobre o dataset atual
- analisar_variavel_especifica: Analisa uma variável/coluna específica em detalhes
- analisar_tendencias_temporais: Analisa padrões temporais nos dados
- detectar_clusters: Identifica agrupamentos usando K-means
- resposta_direta: Responde perguntas diretas e específicas

COMPORTAMENTO PARA PERGUNTAS ESPECÍFICAS:
- "Sobre o que é a tabela?" → Use obter_contexto_atual
- "Qual a média da coluna X?" → Use resposta_direta
- "Quais outliers da coluna Y?" → Use analisar_variavel_especifica
- "Analise a variável Z" → Use analisar_variavel_especifica
- "Detecte clusters" → Use detectar_clusters
- "Tendências temporais" → Use analisar_tendencias_temporais
- "Crie gráficos" → Use criar_grafico_automatico

ADAPTAÇÃO POR TIPO DE DADOS:

📊 DADOS DE FRAUDE (Class, V1-V28, Amount):
- Linguagem: "transações", "fraudes", "desbalanceamento"
- Foco: desbalanceamento, outliers, padrões fraudulentos
- Gráficos: distribuição de classes, comparação normal/fraude

🏪 DADOS DE VENDAS (sales, price, revenue, product):
- Linguagem: "vendas", "produtos", "receita", "performance comercial"
- Foco: performance, produtos top, análise de receita
- Gráficos: histogramas vendas, ranking produtos

🔬 DADOS CIENTÍFICOS (species, petal, sepal, length):
- Linguagem: "espécies", "medidas", "classificação"
- Foco: classificação, correlações entre medidas
- Gráficos: scatter plots por classe, distribuições

🎯 DADOS GERAIS:
- Linguagem: "variáveis", "correlações", "distribuições"
- Foco: estatísticas descritivas, correlações
- Gráficos: matriz correlação, distribuições

FLUXO OBRIGATÓRIO:
1. Para análise completa: carregar_csv → analisar_automaticamente → criar_grafico_automatico
2. Para perguntas específicas: usar ferramenta apropriada diretamente
3. SEMPRE adapte linguagem ao tipo de dados detectado
4. SEMPRE explique suas decisões

IMPORTANTE:
- Use tipo_analise="auto" para gráficos adaptativos
- Cada análise cria UM gráfico único (grafico_atual.png)
- Adapte completamente sua linguagem ao contexto dos dados

Responda sempre de forma clara, direta e totalmente adaptada ao contexto dos dados."""),
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Criar o agente
    agente = create_tool_calling_agent(llm, ferramentas, prompt)
    
    # Criar o executor
    executor = AgentExecutor(
        agent=agente,
        tools=ferramentas,
        memory=memoria,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=int(os.getenv('AGENT_MAX_ITERATIONS', '10'))
    )
    
    return executor

# Criar o agente
agente_eda = criar_agente_eda()
print("🤖 Agente EDA LangChain criado com sucesso!")

# Função para interagir com o agente
def perguntar_ao_agente(pergunta: str) -> str:
    """💬 Faz uma pergunta ao agente EDA"""
    try:
        print(f"\n🤔 PERGUNTA: {pergunta}")
        print("🧠 AGENTE PENSANDO...\n")
        
        resposta = agente_eda.invoke({"input": pergunta})
        return resposta['output']
        
    except Exception as e:
        return f"❌ ERRO: {str(e)}"

print("💬 Sistema de perguntas configurado!")
print("\n" + "="*60)
print("🎯 AGENTE EDA PREMIUM PRONTO PARA USO!")
print("="*60)
# agente_eda.py CORRIGIDO - PARTE 8: TESTES E FINALIZAÇÃO

# ===== TESTE BÁSICO DO SISTEMA =====

def teste_sistema_basico():
    """🧪 Teste básico para verificar se o sistema está funcionando"""
    print("\n🧪 EXECUTANDO TESTE BÁSICO DO SISTEMA...")
    
    # Verificar se as ferramentas estão disponíveis
    ferramentas_disponiveis = [
        carregar_csv, 
        analisar_automaticamente, 
        criar_grafico_automatico, 
        obter_contexto_atual, 
        analisar_variavel_especifica, 
        analisar_tendencias_temporais, 
        detectar_clusters, 
        resposta_direta
    ]
    print(f"✅ {len(ferramentas_disponiveis)} ferramentas carregadas")
    
    # Verificar se o agente foi criado
    if agente_eda:
        print("✅ Agente EDA criado com sucesso")
    else:
        print("❌ Erro na criação do agente")
    
    # Verificar variáveis globais
    print(f"✅ Variáveis globais: dataset_atual={type(dataset_atual)}, descobertas={len(descobertas_memoria)}")
    
    # Verificar LLM
    if llm:
        print("✅ LLM configurado corretamente")
    else:
        print("❌ Problema na configuração do LLM")
    
    print("🎯 TESTE BÁSICO CONCLUÍDO!\n")

# Executar teste básico
teste_sistema_basico()

# ===== INFORMAÇÕES DO SISTEMA FINAL =====

print("📋 INFORMAÇÕES DO SISTEMA FINAL:")
print("="*60)
print("🔧 FERRAMENTAS DISPONÍVEIS (8 TOTAL):")
print("   1. carregar_csv - Carrega e analisa qualquer CSV")
print("   2. analisar_automaticamente - Análise completa automática")
print("   3. criar_grafico_automatico - Gera gráficos adaptativos")
print("   4. obter_contexto_atual - Informa sobre tabela atual")
print("   5. analisar_variavel_especifica - Análise granular de colunas")
print("   6. analisar_tendencias_temporais - Análise de séries temporais")
print("   7. detectar_clusters - Identifica agrupamentos (K-means)")
print("   8. resposta_direta - Responde perguntas específicas")
print("")
print("🧠 TIPOS DE DADOS SUPORTADOS:")
print("   🚨 Fraude/Segurança - Desbalanceamento e outliers")
print("   🏪 Vendas/Comercial - Performance e sazonalidade")
print("   👥 RH/Recursos Humanos - Equidade e demographics")
print("   🔬 Científico/Experimental - Classificações e correlações")
print("   📅 Temporal/Séries - Tendências e sazonalidade")
print("   🎯 Geral - Estatísticas descritivas abrangentes")
print("")
print("💬 COMO USAR:")
print("   from agente_eda import perguntar_ao_agente")
print("   resposta = perguntar_ao_agente('Carregue o arquivo meus_dados.csv')")
print("")
print("📊 CAPACIDADES DE Q&A ESPECÍFICO:")
print("   - 'Sobre o que é a tabela?' → Contexto completo")
print("   - 'Qual a média da coluna X?' → Resposta direta")
print("   - 'Quais outliers da coluna Y?' → Detecção IQR")
print("   - 'Analise a variável Z' → Análise granular")
print("   - 'Detecte clusters' → K-means automático")
print("   - 'Tendências temporais' → Análise de séries")
print("")
print("🎯 STATUS: AGENTE EDA UNIVERSAL COM Q&A ESPECÍFICO!")
print("="*60)

# ===== FUNÇÃO DE RESET MELHORADA =====

def resetar_agente():
    """🔄 Reseta o agente para nova análise"""
    global dataset_atual, descobertas_memoria
    
    dataset_atual = None
    descobertas_memoria = []
    
    # Limpar APENAS o gráfico atual
    if os.path.exists('grafico_atual.png'):
        try:
            os.remove('grafico_atual.png')
            print("🗑️ Gráfico anterior removido")
        except:
            pass
    
    print("✅ Agente resetado com sucesso!")
    return True

# ===== FUNÇÃO PRINCIPAL DE USO =====

def usar_agente(pergunta: str = None):
    """🎯 Função principal para usar o agente"""
    
    if pergunta:
        return perguntar_ao_agente(pergunta)
    else:
        print("\n🎯 AGENTE EDA UNIVERSAL ATIVO!")
        print("💬 Use: perguntar_ao_agente('sua pergunta')")
        print("🔄 Para resetar: resetar_agente()")
        print("")
        print("📚 EXEMPLOS DE PERGUNTAS:")
        print("   • Carregue o arquivo data/creditcard.csv")
        print("   • Sobre o que é esta tabela?")
        print("   • Qual a média da coluna Amount?")
        print("   • Quais outliers da coluna Sales?")
        print("   • Detecte agrupamentos nos dados")
        print("   • Analise tendências temporais")
        
        return "Agente pronto para uso!"

# ===== STATUS FINAL =====

print("\n" + "🎉" * 20)
print("🏆 AGENTE EDA UNIVERSAL FINALIZADO!")
print("✅ 8 ferramentas especializadas")
print("📊 Q&A específico para qualquer pergunta EDA")
print("🧠 Detecção automática de tipos de dados")
print("🎨 Gráfico único por análise (grafico_atual.png)")
print("💬 Interface conversacional com memória")
print("🔄 Sistema de reset otimizado")
print("🎯 PRONTO PARA ENTREGA E AVALIAÇÃO!")
print("🎉" * 20)

# ===== DEMONSTRAÇÃO RÁPIDA =====

def demo_final():
    """🚀 Demonstração final do agente"""
    print("\n🚀 DEMONSTRAÇÃO FINAL:")
    print("="*40)
    
    exemplos = [
        "Carregue o arquivo data/creditcard.csv",
        "Sobre o que é esta tabela?",
        "Qual a média da coluna Amount?",
        "Detecte agrupamentos nos dados",
        "Quais outliers da coluna Amount?"
    ]
    
    print("📚 TESTE ESTAS PERGUNTAS:")
    for i, exemplo in enumerate(exemplos, 1):
        print(f"{i}. perguntar_ao_agente('{exemplo}')")
    
    print("\n💡 CAPACIDADES FINAIS:")
    print("   🎯 Responde qualquer pergunta sobre EDA")
    print("   📊 Cria gráficos específicos por tipo de dados")
    print("   🧠 Mantém contexto entre perguntas")
    print("   🔄 Funciona com qualquer CSV")
    print("   💬 Interface conversacional natural")
    print("="*40)

demo_final()

# Se executado diretamente, mostrar menu
if __name__ == "__main__":
    print("\n🚀 MENU DE OPÇÕES:")
    print("1. usar_agente() - Instruções de uso")
    print("2. resetar_agente() - Limpar dados anteriores")  
    print("3. demo_final() - Ver demonstração final")
    print("\n💬 PARA USAR:")
    print("   perguntar_ao_agente('Carregue o arquivo data/creditcard.csv')")
    print("   perguntar_ao_agente('Sobre o que é esta tabela?')")
    print("   perguntar_ao_agente('Qual a média da coluna Amount?')")
    print("\n📊 ARQUIVO DE GRÁFICO ÚNICO:")
    print("   - Cada análise substitui: grafico_atual.png")
    print("   - Dashboard sempre mostra o gráfico da análise atual")
    print("\n🎯 AGENTE PRONTO PARA USO!")