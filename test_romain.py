import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Chargement du dataset
data_budget = pd.read_csv('dataset_marketing_total.csv')

# Préparation des données pour la régression linéaire
X = data_budget[['Budget Total']].values  # Sélection correcte de la colonne pour la variable indépendante
Y = data_budget['Retour sur ventes Mix Media'].values  # Sélection correcte de la colonne pour la variable dépendante

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, Y)

# Calcul des prédictions pour tracer la ligne de régression
Y_pred = model.predict(X)

# Affichage des données et de la ligne de régression
plt.scatter(X, Y, color='blue', marker='o', s=10, label='Données réelles')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Ligne de régression')
plt.title('Régression Linéaire entre le budget total de communication et le retour sur investissement')
plt.xlabel('Budget')
plt.ylabel('Retour sur ventes')
plt.legend()
plt.show()