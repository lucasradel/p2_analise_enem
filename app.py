import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, inspect
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="ENEM - Análise Sociodemográfica",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CONEXÃO E CARREGAMENTO AUTOMÁTICO DE DADOS
# ============================================

@st.cache_resource(show_spinner=False)
def connect_to_database():
    """Conecta ao banco de dados automaticamente"""
    try:
        connection_string = "postgresql://data_iesb:iesb@bigdata.dataiesb.com/iesb"
        engine = create_engine(connection_string)
        return engine
    except:
        return None

@st.cache_data(show_spinner=False)
def load_all_data():
    """Carrega todas as tabelas automaticamente"""
    engine = connect_to_database()
    if engine is None:
        return {}
    
    try:
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()
        
        # Mapeamento automático das tabelas
        table_mapping = {}
        
        # Buscar tabelas por padrões conhecidos
        for table in all_tables:
            table_lower = table.lower()
            
            if any(x in table_lower for x in ['populacao', 'censo', 'demografico']):
                table_mapping['populacao'] = table
            elif any(x in table_lower for x in ['municipio', 'municípios']):
                table_mapping['municipio'] = table
            elif any(x in table_lower for x in ['pib', 'economico']):
                table_mapping['pib'] = table
            elif any(x in table_lower for x in ['educacao', 'ensino', 'escola']):
                table_mapping['educacao'] = table
            elif any(x in table_lower for x in ['enem', 'nota', 'prova']):
                table_mapping['enem'] = table
        
        # Carregar dados
        dataframes = {}
        for name, table_name in table_mapping.items():
            try:
                query = f'SELECT * FROM "{table_name}" LIMIT 2000'
                dataframes[name] = pd.read_sql(query, engine)
                
                # Log para debug
                if not dataframes[name].empty:
                    st.sidebar.success(f"✅ {table_name}: {dataframes[name].shape[0]} registros")
            except Exception as e:
                st.sidebar.error(f"❌ Erro em {table_name}")
                dataframes[name] = pd.DataFrame()
        
        return dataframes
    
    except Exception as e:
        st.sidebar.error(f"Erro geral: {str(e)}")
        return {}

# ============================================
# INICIALIZAÇÃO DOS DADOS
# ============================================

# Carregar dados automaticamente no início
if 'data_loaded' not in st.session_state:
    with st.spinner("🔄 Carregando dados do banco de dados..."):
        st.session_state.dataframes = load_all_data()
        st.session_state.data_loaded = True

# ============================================
# FUNÇÕES DE ANÁLISE E VISUALIZAÇÃO
# ============================================

def create_simple_correlation_matrix():
    """Cria uma matriz de correlação simplificada e focada"""
    
    # Coletar todas as variáveis numéricas importantes
    correlation_data = {}
    
    for dataset_name, df in st.session_state.dataframes.items():
        if not df.empty:
            # Pegar apenas colunas numéricas
            numeric_cols = df.select_dtypes(include=[np.number])
            
            # Filtrar apenas colunas importantes (evitar IDs e códigos)
            important_cols = []
            for col in numeric_cols.columns:
                col_lower = col.lower()
                # Excluir colunas que são provavelmente IDs ou códigos
                if not any(exclude in col_lower for exclude in ['id', 'cod', 'chave', 'key', 'index']):
                    # Priorizar colunas com nomes significativos
                    if any(keyword in col_lower for keyword in ['nota', 'media', 'ideb', 'renda', 'pib', 'populacao', 'matricula', 'docente', 'escola']):
                        important_cols.append(col)
                    elif len(numeric_cols[col].unique()) > 10:  # Evitar variáveis categóricas numéricas
                        important_cols.append(col)
            
            # Limitar a 15 colunas por dataset para não poluir
            important_cols = important_cols[:15]
            
            if important_cols:
                # Adicionar prefixo para identificar origem
                df_filtered = df[important_cols].copy()
                df_filtered.columns = [f"{dataset_name}_{col}" for col in df_filtered.columns]
                
                # Adicionar ao dicionário de correlação
                for col in df_filtered.columns:
                    correlation_data[col] = df_filtered[col]
    
    if not correlation_data:
        return None
    
    # Criar DataFrame de correlação
    corr_df = pd.DataFrame(correlation_data)
    
    # Calcular matriz de correlação
    corr_matrix = corr_df.corr()
    
    # Filtrar para mostrar apenas correlações fortes
    strong_corr_threshold = 0.3
    mask = np.abs(corr_matrix) > strong_corr_threshold
    corr_matrix_filtered = corr_matrix.where(mask)
    
    # Ordenar por similaridade de correlação para melhor visualização
    corr_matrix_sorted = corr_matrix_filtered.fillna(0)
    
    return corr_matrix_sorted

def create_top_correlations():
    """Identifica as principais correlações entre variáveis"""
    
    all_data = []
    
    for dataset_name, df in st.session_state.dataframes.items():
        if not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number])
            
            # Filtrar colunas relevantes
            relevant_cols = []
            for col in numeric_cols.columns:
                col_lower = col.lower()
                if not any(exclude in col_lower for exclude in ['id', 'cod', 'chave', 'key', 'index', 'ano']):
                    if len(numeric_cols[col].unique()) > 5:  # Evitar variáveis com poucos valores únicos
                        relevant_cols.append(col)
            
            relevant_cols = relevant_cols[:10]  # Limitar a 10 colunas por dataset
            
            if relevant_cols:
                df_filtered = df[relevant_cols].copy()
                df_filtered.columns = [f"{dataset_name[:3]}_{col[:20]}" for col in df_filtered.columns]
                all_data.append(df_filtered)
    
    if not all_data:
        return None
    
    # Combinar dados
    combined_df = pd.concat(all_data, axis=1)
    
    # Calcular matriz de correlação
    corr_matrix = combined_df.corr()
    
    # Extrair pares de correlação fortes
    correlations = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.5:  # Apenas correlações fortes
                correlations.append({
                    'Variável 1': corr_matrix.columns[i],
                    'Variável 2': corr_matrix.columns[j],
                    'Correlação': corr_value,
                    'Tipo': 'Forte Positiva' if corr_value > 0.7 else 
                           'Moderada Positiva' if corr_value > 0.3 else
                           'Forte Negativa' if corr_value < -0.7 else
                           'Moderada Negativa' if corr_value < -0.3 else 'Fraca'
                })
    
    # Ordenar por força da correlação
    correlations_df = pd.DataFrame(correlations)
    if not correlations_df.empty:
        correlations_df = correlations_df.sort_values('Correlação', key=abs, ascending=False)
    
    return correlations_df

def create_focused_correlation_plot():
    """Cria um gráfico de correlação focado nas variáveis mais importantes"""
    
    # Identificar variáveis principais por dataset
    key_variables = {}
    
    for dataset_name, df in st.session_state.dataframes.items():
        if not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number])
            
            # Procurar variáveis-chave por nome
            key_vars = []
            for col in numeric_cols.columns:
                col_lower = col.lower()
                
                # Variáveis prioritárias
                if any(keyword in col_lower for keyword in ['nota', 'media', 'score', 'desempenho']):
                    key_vars.append((col, 3))  # Alta prioridade
                elif any(keyword in col_lower for keyword in ['ideb', 'renda', 'pib', 'populacao']):
                    key_vars.append((col, 2))  # Média prioridade
                elif any(keyword in col_lower for keyword in ['matricula', 'docente', 'escola', 'taxa']):
                    key_vars.append((col, 1))  # Baixa prioridade
            
            # Ordenar por prioridade e pegar as top 5
            key_vars.sort(key=lambda x: x[1], reverse=True)
            key_vars = [var[0] for var in key_vars[:5]]
            
            if key_vars:
                key_variables[dataset_name] = key_vars
    
    # Criar DataFrame combinado com variáveis-chave
    combined_data = {}
    
    for dataset_name, vars_list in key_variables.items():
        df = st.session_state.dataframes[dataset_name]
        for var in vars_list:
            if var in df.columns:
                combined_data[f"{dataset_name[:3]}_{var[:15]}"] = df[var]
    
    if len(combined_data) < 2:
        return None
    
    corr_df = pd.DataFrame(combined_data)
    
    # Calcular correlação
    corr_matrix = corr_df.corr()
    
    # Criar heatmap
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlação"),
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        aspect="auto"
    )
    
    fig.update_layout(
        title="Correlações entre Variáveis-Chave",
        height=500,
        xaxis_title="Variáveis",
        yaxis_title="Variáveis"
    )
    
    return fig

def create_correlation_network():
    """Cria um gráfico de rede de correlações"""
    
    # Pegar top correlações
    correlations_df = create_top_correlations()
    
    if correlations_df is None or correlations_df.empty:
        return None
    
    # Limitar a 20 correlações mais fortes
    top_correlations = correlations_df.head(20)
    
    # Criar grafo de rede
    nodes = set()
    edges = []
    
    for _, row in top_correlations.iterrows():
        nodes.add(row['Variável 1'])
        nodes.add(row['Variável 2'])
        edges.append({
            'source': row['Variável 1'],
            'target': row['Variável 2'],
            'value': abs(row['Correlação']),
            'correlation': row['Correlação']
        })
    
    # Criar figura
    fig = go.Figure()
    
    # Posicionar nós em círculo
    nodes_list = list(nodes)
    num_nodes = len(nodes_list)
    
    for i, node in enumerate(nodes_list):
        angle = 2 * np.pi * i / num_nodes
        x = np.cos(angle)
        y = np.sin(angle)
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            text=[node[:15]],
            textposition="bottom center",
            marker=dict(size=20, color='lightblue'),
            name=node,
            hoverinfo='text',
            hovertext=f"Variável: {node}"
        ))
    
    # Adicionar arestas (conexões)
    for edge in edges:
        source_idx = nodes_list.index(edge['source'])
        target_idx = nodes_list.index(edge['target'])
        
        source_angle = 2 * np.pi * source_idx / num_nodes
        target_angle = 2 * np.pi * target_idx / num_nodes
        
        source_x = np.cos(source_angle)
        source_y = np.sin(source_angle)
        target_x = np.cos(target_angle)
        target_y = np.sin(target_angle)
        
        # Linha mais grossa para correlações mais fortes
        line_width = edge['value'] * 5
        
        # Cor diferente para positiva/negativa
        line_color = 'green' if edge['correlation'] > 0 else 'red'
        
        fig.add_trace(go.Scatter(
            x=[source_x, target_x, None],
            y=[source_y, target_y, None],
            mode='lines',
            line=dict(width=line_width, color=line_color),
            hoverinfo='text',
            hovertext=f"Correlação: {edge['correlation']:.3f}",
            showlegend=False
        ))
    
    fig.update_layout(
        title="Rede de Correlações (Top 20)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# ============================================
# PÁGINAS DA APLICAÇÃO
# ============================================

def show_introduction():
    """Página de introdução"""
    st.title("📚 Análise dos Fatores Sociodemográficos do ENEM")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Questão de Pesquisa")
        st.markdown("""
        ### **Quais os fatores sociodemográficos estão associados ao desempenho dos estudantes no ENEM nos municípios brasileiros?**
        
        Esta análise visa identificar e explorar os principais fatores demográficos, 
        econômicos e sociais que influenciam o desempenho educacional no Brasil.
        
        ### **Objetivos Específicos:**
        1. Identificar correlações entre indicadores sociodemográficos
        2. Analisar padrões nos dados educacionais
        3. Explorar relações entre diferentes dimensões (demografia, educação)
        4. Visualizar insights através de análise estatística
        """)
    
    with col2:
        st.header("📊 Status dos Dados")
        
        if 'dataframes' in st.session_state:
            total_rows = 0
            for name, df in st.session_state.dataframes.items():
                if not df.empty:
                    total_rows += df.shape[0]
                    st.success(f"✅ **{name.upper()}**")
                    st.caption(f"  {df.shape[0]:,} registros | {df.shape[1]} colunas")
                else:
                    st.warning(f"⚠️ {name.capitalize()} (vazio)")
            
            st.metric("Total de Dados", f"{total_rows:,} registros")
        
        st.header("📋 Metodologia")
        st.markdown("""
        - **Análise Exploratória**: Estatísticas descritivas
        - **Correlação**: Relações entre variáveis
        - **Visualização**: Gráficos interativos
        - **Interpretação**: Insights baseados em dados
        """)
    
    st.markdown("---")
    
    st.header("🔍 Estrutura da Análise")
    
    cols = st.columns(3)
    with cols[0]:
        st.subheader("1. Introdução")
        st.markdown("""
        Contexto da pesquisa
        Objetivos e metodologia
        Visão geral dos dados
        """)
    
    with cols[1]:
        st.subheader("2. Dashboard")
        st.markdown("""
        Visualizações principais
        Estatísticas descritivas
        Insights iniciais
        """)
    
    with cols[2]:
        st.subheader("3. Correlações")
        st.markdown("""
        Análise estatística
        Relações entre variáveis
        Padrões identificados
        """)
    
    st.markdown("---")
    
    st.header("🚀 Como Usar")
    st.info("""
    **Navegação:**
    1. Use o menu lateral para escolher entre Introdução, Dashboard ou Correlações
    2. Os dados são carregados automaticamente ao abrir a aplicação
    3. Clique em 'Recarregar Dados' se necessário
    
    **Análise:**
    - Explore os gráficos interativos
    - Observe as correlações entre variáveis
    - Identifique padrões nos dados
    """)

def show_dashboard():
    """Página com visualizações principais"""
    st.title("📊 Dashboard de Análise")
    
    # Verificar se há dados carregados
    if not st.session_state.dataframes or all(df.empty for df in st.session_state.dataframes.values()):
        st.error("❌ Nenhum dado foi carregado do banco de dados.")
        return
    
    # Seção 1: Visão Geral dos Dados
    st.header("📈 Visão Geral dos Dados")
    
    # Estatísticas rápidas
    cols = st.columns(4)
    stats_data = []
    
    for idx, (name, df) in enumerate(st.session_state.dataframes.items()):
        if not df.empty and idx < 4:
            with cols[idx]:
                numeric_cols = df.select_dtypes(include=[np.number])
                if not numeric_cols.empty:
                    # Pegar primeira coluna numérica significativa
                    first_col = numeric_cols.columns[0]
                    mean_val = df[first_col].mean()
                    
                    # Nome amigável para exibição
                    display_name = name.capitalize().replace('_', ' ')
                    
                    st.metric(
                        label=f"Média em {display_name}",
                        value=f"{mean_val:,.1f}",
                        delta=f"Base: {df.shape[0]:,} registros"
                    )
                    stats_data.append((name, df.shape[0], df.shape[1]))
    
    # Seção 2: Distribuição das Variáveis Principais
    st.header("📊 Distribuição das Variáveis")
    
    # Selecionar dataset para análise
    available_datasets = [name for name, df in st.session_state.dataframes.items() if not df.empty]
    
    if available_datasets:
        selected_dataset = st.selectbox("Selecione o dataset para análise:", available_datasets)
        
        df_selected = st.session_state.dataframes[selected_dataset]
        numeric_cols = df_selected.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Limitar a 5 colunas para não poluir
            numeric_cols = numeric_cols[:5]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Histograma")
                selected_var = st.selectbox("Selecione a variável:", numeric_cols)
                
                fig = px.histogram(
                    df_selected, 
                    x=selected_var,
                    title=f"Distribuição de {selected_var}",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Box Plot")
                selected_var_box = st.selectbox("Selecione a variável para box plot:", 
                                              numeric_cols, key="box_var")
                
                fig = px.box(
                    df_selected,
                    y=selected_var_box,
                    title=f"Box Plot de {selected_var_box}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"O dataset {selected_dataset} não possui variáveis numéricas para análise.")
    else:
        st.warning("Nenhum dataset disponível para análise.")
    
    # Seção 3: Insights Iniciais
    st.header("💡 Insights Iniciais")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        with st.expander("📋 Dados Carregados", expanded=True):
            for name, df in st.session_state.dataframes.items():
                if not df.empty:
                    st.write(f"**{name.upper()}**:")
                    st.write(f"- Registros: {df.shape[0]:,}")
                    st.write(f"- Colunas: {df.shape[1]}")
                    st.write(f"- Colunas numéricas: {df.select_dtypes(include=[np.number]).shape[1]}")
                    st.write("---")
    
    with insights_col2:
        with st.expander("🔍 Variáveis Disponíveis", expanded=True):
            for name, df in st.session_state.dataframes.items():
                if not df.empty:
                    st.write(f"**{name}**:")
                    
                    # Mostrar algumas colunas como exemplo
                    sample_cols = df.columns[:5].tolist()
                    for col in sample_cols:
                        st.write(f"  - {col}")
                    
                    if len(df.columns) > 5:
                        st.write(f"  ... e mais {len(df.columns) - 5} colunas")
                    st.write("---")

def show_correlations():
    """Página de análise de correlações simplificada"""
    st.title("🔗 Análise de Correlações")
    
    # Verificar se há dados suficientes
    numeric_count = 0
    for df in st.session_state.dataframes.values():
        if not df.empty:
            numeric_count += df.select_dtypes(include=[np.number]).shape[1]
    
    if numeric_count < 2:
        st.warning("⚠️ Dados insuficientes para análise de correlação.")
        st.info("É necessário pelo menos 2 variáveis numéricas para calcular correlações.")
        return
    
    st.markdown("""
    ### **O que é análise de correlação?**
    
    A correlação mede a relação entre duas variáveis. Valores próximos de:
    - **+1**: Correlação positiva forte (quando uma aumenta, a outra também aumenta)
    - **0**: Sem correlação
    - **-1**: Correlação negativa forte (quando uma aumenta, a outra diminui)
    """)
    
    # Seção 1: Heatmap de Correlação Simplificado
    st.header("📊 Mapa de Calor das Correlações")
    
    correlation_fig = create_focused_correlation_plot()
    
    if correlation_fig:
        st.plotly_chart(correlation_fig, use_container_width=True)
        
        with st.expander("📝 Interpretação do Mapa de Calor"):
            st.markdown("""
            **Como interpretar:**
            - **Cores quentes (vermelho)**: Correlação positiva
            - **Cores frias (azul)**: Correlação negativa
            - **Intensidade da cor**: Força da correlação
            
            **Padrões para observar:**
            1. Blocos de cores similares indicam grupos de variáveis relacionadas
            2. Correlações fortes (vermelho/azul intenso) merecem atenção especial
            3. Ausência de padrão (cores próximas ao branco) sugere independência
            """)
    else:
        st.info("Carregando análise de correlações...")
    
    # Seção 2: Top Correlações
    st.header("🏆 Principais Correlações Identificadas")
    
    top_correlations = create_top_correlations()
    
    if top_correlations is not None and not top_correlations.empty:
        # Mostrar as top 10 correlações
        st.dataframe(
            top_correlations.head(10),
            use_container_width=True,
            column_config={
                "Variável 1": st.column_config.TextColumn("Variável 1", width="medium"),
                "Variável 2": st.column_config.TextColumn("Variável 2", width="medium"),
                "Correlação": st.column_config.NumberColumn(
                    "Correlação",
                    format="%.3f",
                    help="Valor entre -1 e 1"
                ),
                "Tipo": st.column_config.TextColumn("Força da Correlação")
            }
        )
        
        # Análise das correlações mais fortes
        if not top_correlations.empty:
            strongest_pos = top_correlations[top_correlations['Correlação'] > 0].iloc[0] if any(top_correlations['Correlação'] > 0) else None
            strongest_neg = top_correlations[top_correlations['Correlação'] < 0].iloc[-1] if any(top_correlations['Correlação'] < 0) else None
            
            col1, col2 = st.columns(2)
            
            with col1:
                if strongest_pos is not None:
                    st.metric(
                        label="📈 Correlação Positiva Mais Forte",
                        value=f"{strongest_pos['Correlação']:.3f}",
                        delta=f"{strongest_pos['Variável 1'][:20]} ↔ {strongest_pos['Variável 2'][:20]}"
                    )
            
            with col2:
                if strongest_neg is not None:
                    st.metric(
                        label="📉 Correlação Negativa Mais Forte",
                        value=f"{strongest_neg['Correlação']:.3f}",
                        delta=f"{strongest_neg['Variável 1'][:20]} ↔ {strongest_neg['Variável 2'][:20]}"
                    )
    else:
        st.info("Nenhuma correlação forte encontrada nos dados atuais.")
    
    # Seção 3: Análise Detalhada de Correlação Específica
    st.header("🔍 Análise Detalhada de Correlação")
    
    # Coletar todas as variáveis numéricas disponíveis
    all_variables = []
    for dataset_name, df in st.session_state.dataframes.items():
        if not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number])
            for col in numeric_cols.columns:
                # Filtrar colunas não relevantes
                col_lower = col.lower()
                if not any(exclude in col_lower for exclude in ['id', 'cod', 'chave', 'key', 'index']):
                    if len(df[col].unique()) > 5:
                        all_variables.append({
                            'name': f"{dataset_name[:3]}_{col[:20]}",
                            'dataset': dataset_name,
                            'column': col
                        })
    
    if len(all_variables) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            var1_option = st.selectbox(
                "Selecione a primeira variável:",
                [f"{v['dataset']}: {v['column']}" for v in all_variables],
                key="var1_select"
            )
        
        with col2:
            # Filtrar segunda variável (não pode ser a mesma)
            var1_idx = [f"{v['dataset']}: {v['column']}" for v in all_variables].index(var1_option)
            other_vars = [f"{v['dataset']}: {v['column']}" for i, v in enumerate(all_variables) if i != var1_idx]
            
            var2_option = st.selectbox(
                "Selecione a segunda variável:",
                other_vars,
                key="var2_select"
            )
        
        if st.button("🔍 Analisar Correlação", type="primary"):
            # Extrair dados das variáveis selecionadas
            var1_data = None
            var2_data = None
            
            for v in all_variables:
                current_var = f"{v['dataset']}: {v['column']}"
                if current_var == var1_option:
                    var1_data = st.session_state.dataframes[v['dataset']][v['column']]
                if current_var == var2_option:
                    var2_data = st.session_state.dataframes[v['dataset']][v['column']]
            
            if var1_data is not None and var2_data is not None:
                # Calcular correlação
                valid_data = pd.concat([var1_data, var2_data], axis=1).dropna()
                
                if len(valid_data) >= 2:
                    correlation = np.corrcoef(valid_data.iloc[:, 0], valid_data.iloc[:, 1])[0, 1]
                    
                    # Exibir resultado
                    st.subheader("Resultado da Análise")
                    
                    col_result1, col_result2, col_result3 = st.columns(3)
                    
                    with col_result1:
                        st.metric("Correlação", f"{correlation:.3f}")
                    
                    with col_result2:
                        if abs(correlation) > 0.7:
                            strength = "Muito Forte"
                        elif abs(correlation) > 0.5:
                            strength = "Forte"
                        elif abs(correlation) > 0.3:
                            strength = "Moderada"
                        else:
                            strength = "Fraca"
                        st.metric("Força", strength)
                    
                    with col_result3:
                        direction = "Positiva" if correlation > 0 else "Negativa"
                        st.metric("Direção", direction)
                    
                    # Scatter plot
                    fig = px.scatter(
                        x=valid_data.iloc[:, 0],
                        y=valid_data.iloc[:, 1],
                        trendline="ols",
                        labels={'x': var1_option, 'y': var2_option},
                        title=f"Relação entre {var1_option[:30]} e {var2_option[:30]}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretação
                    with st.expander("📝 Interpretação", expanded=True):
                        if correlation > 0.7:
                            st.success(f"**Correlação positiva muito forte**: As variáveis tendem a aumentar juntas.")
                        elif correlation > 0.3:
                            st.info(f"**Correlação positiva moderada**: Há uma tendência de relação positiva.")
                        elif correlation > -0.3:
                            st.warning(f"**Correlação fraca**: Pouca ou nenhuma relação evidente.")
                        elif correlation > -0.7:
                            st.info(f"**Correlação negativa moderada**: Há uma tendência de relação inversa.")
                        else:
                            st.success(f"**Correlação negativa muito forte**: Quando uma aumenta, a outra tende a diminuir.")
                else:
                    st.warning("Dados insuficientes para calcular correlação.")
    
    # Seção 4: Insights das Correlações
    st.header("💡 Insights e Recomendações")
    
    with st.expander("🎯 O que as correlações nos dizem?"):
        st.markdown("""
        ### **Interpretação das Correlações:**
        
        **Correlação ≠ Causação**
        - Uma correlação forte não significa que uma variável cause a outra
        - Pode haver fatores externos influenciando ambas
        
        **Padrões Comuns em Dados Educacionais:**
        1. **Recursos x Desempenho**: Mais recursos educacionais frequentemente correlacionam com melhor desempenho
        2. **Socioeconomia x Educação**: Indicadores econômicos costumam correlacionar com indicadores educacionais
        3. **Infraestrutura x Acesso**: Recursos físicos podem correlacionar com acesso à educação
        
        **Recomendações para Análise:**
        - Investigue correlações fortes (> 0.7 ou < -0.7)
        - Considere o contexto das variáveis
        - Procure por padrões consistentes entre diferentes datasets
        """)

# ============================================
# CONFIGURAÇÃO DA NAVEGAÇÃO
# ============================================

# Menu de navegação na sidebar
with st.sidebar:
    st.title("📚 ENEM Análise")
    st.markdown("---")
    
    # Status dos dados
    if 'dataframes' in st.session_state:
        loaded_count = sum(1 for df in st.session_state.dataframes.values() if not df.empty)
        total_rows = sum(df.shape[0] for df in st.session_state.dataframes.values() if not df.empty)
        
        st.success(f"✅ {loaded_count}/5 datasets")
        st.caption(f"📊 {total_rows:,} registros totais")
    else:
        st.warning("⚠️ Aguardando dados...")
    
    st.markdown("---")
    
    st.header("Navegação")
    
    # Opções de navegação simplificadas
    page_options = {
        "🏠 Introdução": show_introduction,
        "📊 Dashboard": show_dashboard,
        "🔗 Correlações": show_correlations
    }
    
    # Seleção da página
    selected_page = st.radio(
        "Selecione a página:",
        list(page_options.keys()),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Informações rápidas
    with st.expander("ℹ️ Sobre", expanded=False):
        st.markdown("""
        **Análise dos Fatores Sociodemográficos do ENEM**
        
        **Fontes de Dados:**
        - Banco IESB
        - Dados municipais
        - Indicadores educacionais
        
        **Técnicas:**
        - Análise exploratória
        - Correlação estatística
        - Visualização de dados
        """)
    
    # Botão para recarregar dados
    if st.button("🔄 Recarregar Dados", type="secondary", use_container_width=True):
        with st.spinner("Recarregando dados..."):
            st.session_state.dataframes = load_all_data()
        st.success("Dados recarregados!")
        st.rerun()

# ============================================
# EXIBIÇÃO DA PÁGINA SELECIONADA
# ============================================

# Executar a função da página selecionada
if selected_page in page_options:
    page_options[selected_page]()
else:
    show_introduction()

# ============================================
# RODAPÉ
# ============================================

st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[0]:
    st.caption("📊 **Análise ENEM Sociodemográfica**")
with footer_cols[1]:
    st.caption("🎯 **IESB - Ciência de Dados**")
with footer_cols[2]:
    st.caption("🔄 Dados atualizados automaticamente")