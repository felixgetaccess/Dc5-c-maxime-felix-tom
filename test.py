import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Chargement du dataset
data = pd.read_csv('BudgetRS-CDV.csv')

# Nettoyage des données : suppression des valeurs négatives et de la valeur maximale extrême pour le coût de vente unitaire mix média
data_clean = data[data['Cout de vente unitaire Mix Media'] > 0]
data_clean_no_max = data_clean[data_clean['Cout de vente unitaire Mix Media'] < data_clean['Cout de vente unitaire Mix Media'].max()]

# Préparation des données pour la régression linéaire
X = data_clean_no_max[['Repartition budget RS']].values  # Sélection correcte de la colonne pour la variable indépendante
Y = data_clean_no_max['Cout de vente unitaire Mix Media'].values  # Sélection correcte de la colonne pour la variable dépendante

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, Y)

# Calcul des prédictions pour tracer la ligne de régression
Y_pred = model.predict(X)

# Affichage des données et de la ligne de régression
plt.scatter(X, Y, color='blue', marker='o', s=10, label='Données réelles')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Ligne de régression')
plt.title('Régression Linéaire entre le Coût de vente unitaire Mix Media et la part d\'investissement publicitaire sur les réseaux sociaux')
plt.xlabel('Répartition budget RS')
plt.ylabel('Coût de vente unitaire Mix Media')
plt.legend()
plt.show()