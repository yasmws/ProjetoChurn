# 🔄 ProjetoChurn - Detecção de Churn de Clientes

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Descrição do Projeto

Este projeto implementa modelos de **machine learning** e **deep learning** para **predição de churn de clientes** em uma empresa de telecomunicações. Utilizamos diferentes abordagens, desde redes neurais tradicionais até arquiteturas Transformer adaptadas para dados tabulares, comparando suas performances na identificação de clientes em risco de cancelamento.

### 🎯 Objetivo

Desenvolver e comparar modelos preditivos para detectar clientes que têm alta probabilidade de cancelar seus serviços de telecomunicações, permitindo ações proativas de retenção.

### 🏆 Modelos Implementados

- **MLP (Multi-Layer Perceptron)**: Redes neurais tradicionais com diferentes arquiteturas
- **STAB Transformer**: Arquitetura Transformer adaptada para dados tabulares
- **TabPFN**: Prior-Data Fitted Networks para classificação tabular
- **Random Forest**: Modelo ensemble baseado em árvores de decisão
- **Gradient Boosting**: Modelo ensemble com boosting sequencial
- **XGBoost**: Implementação otimizada de Gradient Boosting

## 👥 Equipe

| Nome | Login | Responsabilidade |
|------|-------|------------------|
| **Brenda Guerra** | `bvga` | Análise Exploratória e Visualizações |
| **Yasmin Maria Wanderley Soares** | `ymws` | Coordenação e Integração |
| **Gabriel Ferreira da Silva** | `gfs4` | Modelos de Deep Learning |
| **Lucas Santiago Monterazo** | `lsm6` | Pré-processamento e Feature Engineering |
| **Matheus Correia** | `mcr` | Modelos de Machine Learning Clássico |

## 📁 Estrutura do Projeto

```text
ProjetoChurn/
├── projetao/
│   └── Churn(2)(1).ipynb                    # Notebook original com modelos de Deep Learning
├── Churn_RebasedPT.ipynb                    # Notebook principal refatorado (português)
├── Churn_RebasedPT_Matheus.ipynb            # Versão com modelos de ML clássico
├── gradient_boosting_best.pkl               # Modelo Gradient Boosting otimizado
├── gradient_boosting_best_without_kfold.pkl # Gradient Boosting sem validação cruzada
├── random_forest_best.pkl                  # Modelo Random Forest otimizado
├── random_forest_best_without_kfold.pkl    # Random Forest sem validação cruzada
├── xgboost_best.pkl                        # Modelo XGBoost otimizado
├── xgboost_best_without_kfold.pkl          # XGBoost sem validação cruzada
├── README.md                               # Este arquivo
├── .gitignore                             # Arquivos ignorados pelo Git
└── requirements.txt                       # Dependências do projeto
```

## 🚀 Como Executar

### 1. **Pré-requisitos**

- Python 3.8 ou superior
- Jupyter Notebook ou Google Colab
- CUDA (opcional, para aceleração GPU)

### 2. **Instalação das Dependências**

```bash
# Clone o repositório
git clone https://github.com/yasmws/ProjetoChurn.git
cd ProjetoChurn

# Instale as dependências
pip install -r requirements.txt

# Ou execute diretamente no notebook (primeira célula já configurada)
```

### 3. **Execução**

```bash
# Notebook principal (versão refatorada em português)
jupyter notebook Churn_RebasedPT.ipynb

# Notebook com modelos de ML clássico
jupyter notebook Churn_RebasedPT_Matheus.ipynb

# Notebook original (versão anterior)
jupyter notebook projetao/Churn\(2\)\(1\).ipynb

# Ou execute no Google Colab (recomendado para acesso a GPU gratuita)
```

### 4. **Dependências Principais**

```python
# Machine Learning e Data Science
tensorflow>=2.8.0
torch>=1.9.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
xgboost>=1.6.0

# Visualização
matplotlib>=3.5.0
seaborn>=0.11.0

# Modelos Especializados
tabpfn>=0.1.0
tabular-transformers>=1.0.0

# Utilitários
kagglehub>=0.1.0
scipy>=1.7.0
joblib>=1.0.0
```

## 📊 Dataset

- **Fonte**: [Kaggle - Customer Churn in Telecom Services](https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services)
- **Tamanho**: ~7,000 registros de clientes
- **Features**: 20+ variáveis incluindo:
  - Dados demográficos (idade, gênero, etc.)
  - Informações de serviços (internet, telefone, streaming)
  - Dados financeiros (charges mensais, total, método de pagamento)
  - **Target**: Churn (Yes/No)

### 🔧 Pré-processamento Aplicado

1. **Limpeza de Dados**: Tratamento de valores faltantes e inconsistências
2. **Feature Engineering**: Criação de variáveis derivadas
3. **Balanceamento**: Oversampling da classe minoritária
4. **Codificação**: One-Hot Encoding para redes neurais, Label Encoding para árvores
5. **Normalização**: StandardScaler para features numéricas

## 📈 Métricas de Avaliação

### Principais Métricas Utilizadas

- **Accuracy**: Proporção geral de acertos
- **Precision**: Proporção de predições positivas corretas
- **Recall**: Proporção de casos positivos identificados
- **F1-Score**: Média harmônica entre precision e recall
- **KS Statistic**: Capacidade de discriminação entre classes (crucial para churn)

### Interpretação para Churn

- **Recall Alto**: Importante para capturar clientes em risco
- **Precision Alta**: Reduz falsos alarmes em campanhas de retenção
- **KS > 0.3**: Considerado bom para separação de classes

## 🛠️ Tecnologias Utilizadas

| Categoria | Tecnologias |
|-----------|-------------|
| **Linguagem** | Python 3.8+ |
| **Deep Learning** | TensorFlow, PyTorch |
| **Machine Learning** | Scikit-learn, XGBoost, TabPFN |
| **Data Science** | Pandas, NumPy, SciPy |
| **Visualização** | Matplotlib, Seaborn |
| **Otimização** | GridSearchCV, Validação Cruzada |
| **Ambiente** | Jupyter Notebook, Google Colab |

## 📚 Referências

- [Dataset Original - Kaggle](https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services)
- [TabPFN Paper](https://arxiv.org/abs/2207.01848)
- [Attention is All You Need - Transformer Paper](https://arxiv.org/abs/1706.03762)
- Documentação TensorFlow e PyTorch
- Slides da disciplina - Prof. Germano Vasconcelos

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 📞 Contato

Para dúvidas ou colaborações, entre em contato com qualquer membro da equipe através dos emails institucionais da UFPE.

---

## 🎓 Sobre o Projeto

Projeto desenvolvido como trabalho final da disciplina de Redes Neurais - UFPE 2025.1
