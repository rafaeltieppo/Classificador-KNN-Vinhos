# -*- coding: utf-8 -*-
"""
PROJETO KNN: CLASSIFICADOR DE VINHOS
Descrição: Classifica vinhos em 'tinto' ou 'branco' baseado em características químicas
"""

# Importação das bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ======================
# 1. CRIANDO O DATASET
# ======================
# Dados fictícios de vinhos (pode substituir por um arquivo CSV)
dados_vinho = {
    'acidez': [7.4, 7.8, 6.3, 8.1, 6.2, 5.9, 7.2, 6.8, 6.5, 7.3],
    'acucar': [1.9, 2.6, 2.3, 2.8, 1.5, 1.2, 2.1, 2.4, 1.8, 2.0],
    'alcool': [9.8, 9.5, 11.1, 10.2, 12.0, 13.1, 10.5, 11.5, 12.5, 10.0],
    'tipo': ['tinto', 'tinto', 'branco', 'branco', 'branco', 'branco', 'tinto', 'branco', 'branco', 'tinto']
}

# Converter para DataFrame
df = pd.DataFrame(dados_vinho)
print("\nDados dos Vinhos:")
print(df.head())

# ======================
# 2. VISUALIZAÇÃO
# ======================
plt.figure(figsize=(10, 6))
plt.scatter(
    df['acidez'], 
    df['alcool'], 
    c=df['tipo'].map({'tinto': 'red', 'branco': 'lightyellow'}),
    s=100,
    edgecolors='black',
    alpha=0.7
)

plt.title('Classificação de Vinhos por Acidez e Teor Alcoólico', fontsize=14)
plt.xlabel('Nível de Acidez (pH)', fontsize=12)
plt.ylabel('Teor Alcoólico (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# ======================
# 3. PREPARAÇÃO DOS DADOS
# ======================
# Selecionando features (X) e target (y)
knn = KNeighborsClassifier(n_neighbors=3)
X = df[['acidez', 'alcool']]  # Características para classificação
y = df['tipo']                 # Rótulos (classes)

# ======================
# 4. TREINANDO O MODELO KNN
# ======================
# Criando o classificador com 3 vizinhos

# Treinando o modelo
knn.fit(X, y)

# ======================
# 5. FAZENDO PREDIÇÕES
# ======================
# Novo vinho para classificação
novo_vinho = {
    'acidez': [7.0, 6.0, 8.0],  
    'alcool': [10.0, 11.5, 13.0]
}

for i in range(len(novo_vinho['acidez'])):
    entrada = [[novo_vinho['acidez'][i], novo_vinho['alcool'][i]]]
    predicao = knn.predict(entrada)
    
    print(f"\nVinho {i+1}:")
    print(f"• Acidez: {entrada[0][0]}")
    print(f"• Álcool: {entrada[0][1]}")
    print(f"► Classificação: {predicao[0].upper()}")

    # Mostrando os vizinhos mais próximos
    distancias, indices = knn.kneighbors(entrada)
    print("\nVizinhos mais próximos usados na classificação:")
    for j in indices[0]:
        print(f"- {df.iloc[j]['tipo']} (Acidez: {df.iloc[j]['acidez']}, Álcool: {df.iloc[j]['alcool']})")

# ======================
# 6. AVALIAÇÃO DO MODELO (SIMPLIFICADA)
# ======================
# Verificando a acurácia nos próprios dados de treino (apenas para demonstração)
acuracia = knn.score(X, y)
print(f"\nAcurácia do modelo nos dados de treino: {acuracia:.2%}")

# ======================
# 7. SALVANDO OS DADOS (OPCIONAL)
# ======================
df.to_csv('dados_vinhos.csv', index=False)
print("\nDataset salvo como 'dados_vinhos.csv'")
