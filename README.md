# Projeto de Predição de Churn - Telecomunicações

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)

## 🎯 Objetivo

Desenvolver modelos preditivos para detectar **churn de clientes** em uma empresa de telecomunicações, utilizando diferentes algoritmos de machine learning e deep learning. Este projeto implementa uma solução completa desde o pré-processamento até a avaliação, com **documentação integralmente em português brasileiro** e comparação sistemática entre múltiplas abordagens.

## 👥 Equipe

| Nome | Login | Responsabilidade |
|------|-------|------------------|
| **Brenda Guerra** | `bvga` | Análise exploratória e visualização dos modelos |
| **Yasmin Maria Wanderley Soares** | `ymws` | Integração, documentação e apresentação |
| **Gabriel Ferreira da Silva** | `gfs4` | Modelagem com MLP e STAB Transformer |
| **Lucas Santiago Monterazo** | `lsm6` | Pré-processamento e engenharia de features |
| **Matheus Correia** | `mcr` | Modelagem com Random Forest e Gradient Boosting |

## 🚀 Como Executar

### 1. **Pré-requisitos**
- Python 3.8 ou superior
- Jupyter Notebook ou Google Colab (recomendado)
- CUDA (opcional, para aceleração GPU)

### 2. **Instalação das Dependências**
```bash
# Clone o repositório
git clone <repository-url>
cd ProjetoChurn

# Instale as dependências principais
pip install tensorflow torch scikit-learn pandas numpy xgboost
pip install matplotlib seaborn tabpfn joblib scipy kagglehub
```

### 3. **Execução Recomendada**

#### 📓 **Notebook Principal (RECOMENDADO)**
```bash
# Abra o notebook principal com documentação completa em PT-BR
jupyter notebook Churn.ipynb
```

**💡 Por que usar este notebook?**
- ✅ **Documentação 100% em português brasileiro**
- ✅ **Pipeline completo de ML desde limpeza até avaliação**
- ✅ **5 experimentações detalhadas com MLP**
- ✅ **Implementação de TabPFN e STAB Transformer**
- ✅ **Funções reutilizáveis bem documentadas**
- ✅ **Visualizações ricas e análises detalhadas**

#### 📊 **Notebooks Alternativos**
```bash
# Versão intermediária
jupyter notebook Churn_RebasedPT.ipynb

# Notebook original com deep learning
jupyter notebook projetao/Churn\(2\)\(1\).ipynb
```

## 🧠 Modelos Implementados

### 🔮 **Redes Neurais e Deep Learning**

#### **MLP (Multi-Layer Perceptron) - 5 Experimentações**
1. **Experimentação 1**: 1 camada oculta (10 neurônios) + Adam + CrossEntropy
2. **Experimentação 2**: 3 camadas ocultas + Dropout (0.3) + RMSprop  
3. **Experimentação 3**: 2 camadas ocultas (20 neurônios) + RMSprop
4. **Experimentação 4**: Ativação tanh + RMSprop + arquitetura customizada
5. **Experimentação 5**: Regularização L2 + Adadelta + early stopping

#### **Modelos Especializados**
- **🤖 STAB Transformer**: Transformer adaptado para dados tabulares
  - Configurações: 2, 4, 8 e 16 attention heads
  - 1 e 2 camadas com dropout configurável
- **⚡ TabPFN**: Prior-Data Fitted Networks para classificação tabular
  - Modelo pré-treinado especializado em dados tabulares

### 🌳 **Modelos Tree-based Clássicos**
- **🌲 Random Forest**: Ensemble de árvores de decisão
- **📈 Gradient Boosting**: Implementação clássica do sklearn  
- **🚀 XGBoost**: Gradient boosting extremo otimizado

##  Tecnologias Utilizadas

### 📚 **Core Machine Learning**
```bash
tensorflow>=2.0.0      # Redes neurais MLP
torch>=1.0.0           # STAB Transformer
scikit-learn>=1.0.0    # Algoritmos clássicos e métricas
tabpfn>=0.1.0          # Modelo pré-treinado tabular
```

### 🚀 **Algoritmos de Ensemble**
```bash
xgboost>=1.5.0         # Gradient boosting otimizado
lightgbm>=3.0.0        # Gradient boosting eficiente  
catboost>=1.0.0        # Especializado em categóricos
```

### 📊 **Data Science Stack**
```bash
pandas>=1.3.0          # Manipulação de dados
numpy>=1.21.0          # Computação numérica
scipy>=1.7.0           # Funções científicas
matplotlib>=3.5.0      # Visualização básica
seaborn>=0.11.0        # Visualização estatística
```

### **Utilitários**
```bash
joblib>=1.1.0          # Persistência de modelos
kagglehub>=0.1.0       # Download de datasets
optuna>=3.0.0          # Otimização de hiperparâmetros
```

## Pipeline de Dados e Funcionalidades

### 🧹 **Pré-processamento Inteligente**
```python
# Funções implementadas no notebook principal:
tratar_dados()                    # Limpeza e tratamento de inconsistências
create_engineered_features()      # Engenharia de features avançada
preprocessing_for_trees()         # Pipeline específico para modelos tree-based
preprocessing_for_neural_nets()   # Pipeline específico para redes neurais
```

### 📈 **Métricas e Avaliação Abrangente**
```python
# Sistema completo de métricas implementado:
calcular_metricas()              # Accuracy, Precision, Recall, F1-Score
calcular_auc_roc()               # Área sob a curva ROC
calcular_ks_statistic()          # Estatística KS para discriminação
plot_roc_curve()                 # Visualização da curva ROC
plot_ks_distribution()           # Distribuições KS para análise
```

### 🎯 **Características Especiais do Projeto**

#### ✨ **Pontos Fortes Únicos**
- **📚 Documentação 100% em PT-BR**: Todo código comentado em português brasileiro
- **🔄 Callbacks Personalizados**: Monitoramento ROC e KS durante treinamento
- **⚖️ Pré-processamento Diferenciado**: Pipelines específicos por tipo de modelo
- **🎨 Visualizações Ricas**: Gráficos informativos e análises detalhadas
- **🔒 Reprodutibilidade**: Seeds fixas e ambiente controlado
- **💾 Persistência**: Modelos salvos automaticamente (.pkl, .h5, .pt)

#### 🔬 **Funcionalidades Avançadas**
- **Early Stopping**: Prevenção de overfitting com parada automática
- **Balanceamento de Classes**: Oversampling inteligente para dados desbalanceados
- **Validação Cruzada**: K-fold para robustez estatística
- **Otimização de Hiperparâmetros**: GridSearch e busca manual
- **Análise de Feature Importance**: Identificação das variáveis mais relevantes

## 📂 Dataset e Métricas de Negócio

### 📊 **Fonte dos Dados**
- **Dataset**: Customer Churn in Telecom Services
- **Origem**: Kaggle via kagglehub
- **Registros**: ~7,000 clientes
- **Features**: 20+ variáveis (demográficas, serviços, financeiras)
- **Target**: Churn binário (Yes/No)

### 🎯 **Métricas de Negócio para Churn**
| Métrica | Importância | Interpretação |
|---------|-------------|---------------|
| **Recall** | 🔥 Crítica | Capturar todos os clientes em risco |
| **Precision** | ⚠️ Alta | Evitar falsos alarmes em campanhas |
| **KS Statistic** | 📈 Essencial | Capacidade de discriminação (>0.3 = bom) |
| **AUC-ROC** | 📊 Importante | Performance geral (>0.8 = excelente) |
| **F1-Score** | ⚖️ Balanceamento | Equilíbrio entre precision e recall |

## 🎯 Resultados e Performance

### **Estrutura de Avaliação Robusta**
- **Divisão Estratificada**: Preserva distribuição de classes
- **Validação Cruzada**: K-fold para estabilidade estatística  
- **Holdout Test**: Conjunto de teste isolado para avaliação final
- **Análise de Overfitting**: Early stopping e regularização

### 🏆 **Modelos Salvos Disponíveis**
Os seguintes modelos otimizados estão salvos e prontos para uso:
- `random_forest_best.pkl` - Melhor Random Forest com validação cruzada
- `gradient_boosting_best.pkl` - Melhor Gradient Boosting com validação cruzada  
- `xgboost_best.pkl` - Melhor XGBoost com validação cruzada
- Versões `*_without_kfold.pkl` - Modelos sem validação cruzada para comparação

## 🎓 Contexto Acadêmico e Contribuições

### 📚 **Objetivo Educacional**
Este projeto foi desenvolvido como trabalho acadêmico com foco em:
- ✅ **Aplicação prática** de técnicas de machine learning e deep learning
- ✅ **Comparação sistemática** entre diferentes algoritmos
- ✅ **Documentação educativa** em português brasileiro
- ✅ **Boas práticas** de desenvolvimento e organização de código
- ✅ **Metodologia científica** na avaliação de modelos


*Projeto desenvolvido para fins acadêmicos - Predição de Churn em Telecomunicações - 2025*
