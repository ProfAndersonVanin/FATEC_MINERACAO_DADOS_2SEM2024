# Importação das Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Carregar os Dados
df = pd.read_csv('caminho_para_o_dataset.csv')  # Substitua pelo caminho do seu arquivo
df = df.dropna()  # Remover valores nulos
df = pd.get_dummies(df, columns=['Fuel Type', 'Transmission'], drop_first=True)  # Codificação de variáveis categóricas

# Visualização do Dataset e análise inicial
print("Informações do Dataset:")
print(df.info())

# Dividir o dataset para tarefas de Classificação, Regressão e Agrupamento
X = df.drop('CO2 Emissions', axis=1)  # Variáveis independentes
y_regressao = df['CO2 Emissions']  # Variável dependente para regressão

# 1. CLASSIFICAÇÃO
# Converter emissões de CO2 em classes (ex.: Baixa, Média, Alta)
df['CO2 Class'] = pd.cut(df['CO2 Emissions'], bins=[0, 150, 250, 400], labels=['Baixa', 'Média', 'Alta'])
y_classificacao = df['CO2 Class']

# Dividir os dados para classificação
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_classificacao, test_size=0.2, random_state=42)

# Usar Decision Tree para classificação
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_class, y_train_class)

# Previsões e avaliação de classificação
y_pred_class = clf.predict(X_test_class)
print("\nAvaliação do Modelo de Classificação:")
print("Acurácia:", accuracy_score(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class))

# 2. REGRESSÃO
# Dividir os dados para regressão
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regressao, test_size=0.2, random_state=42)

# Regressão Linear
linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)
y_pred_linear = linear_model.predict(X_test_reg)

# Avaliação do Modelo de Regressão Linear
print("\nAvaliação do Modelo de Regressão Linear:")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_linear))
print("MSE:", mean_squared_error(y_test_reg, y_pred_linear))
print("R²:", r2_score(y_test_reg, y_pred_linear))

# Regressão com Random Forest para comparação
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_reg, y_train_reg)
y_pred_rf = rf_model.predict(X_test_reg)

# Avaliação do Modelo Random Forest
print("\nAvaliação do Modelo Random Forest Regressor:")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_rf))
print("MSE:", mean_squared_error(y_test_reg, y_pred_rf))
print("R²:", r2_score(y_test_reg, y_pred_rf))

# 3. AGRUPAMENTO (Clustering)
# Padronizar os dados antes do agrupamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Agrupamento com KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Adicionar as informações de clusters ao dataset
df['Cluster'] = clusters

# Redução de dimensionalidade para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel("PCA Componente 1")
plt.ylabel("PCA Componente 2")
plt.title("Agrupamento de Veículos com KMeans")
plt.colorbar(label="Clusters")
plt.show()

# Mostrar alguns exemplos de veículos em cada cluster
for cluster in range(3):
    print(f"\nExemplos do Cluster {cluster}:")
    display(df[df['Cluster'] == cluster].head())
