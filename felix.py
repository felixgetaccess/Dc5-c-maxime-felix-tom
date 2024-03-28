import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Remplacez 'path_to_investments_data' par le chemin réel de votre fichier de données
investments_data = pd.read_excel('path_to_investments_data.xlsx')

# Sélection des colonnes pour les budgets
X_tv = investments_data['Budget Télévision'].values.reshape(-1, 1)
X_social = investments_data['Budget Réseaux Sociaux'].values.reshape(-1, 1)
X_radio = investments_data['Budget Radio'].values.reshape(-1, 1)

# Sélection de la colonne cible pour les ventes
y_sales = investments_data['Ventes'].values

model_tv = LinearRegression()
model_social = LinearRegression()
model_radio = LinearRegression()

model_tv.fit(X_tv, y_sales)
model_social.fit(X_social, y_sales)
model_radio.fit(X_radio, y_sales)

# Gamme de valeurs de budget pour les prédictions
budget_range = np.linspace(0, 40000, 100).reshape(-1, 1)

# Prédictions des ventes
y_pred_tv = model_tv.predict(budget_range)
y_pred_social = model_social.predict(budget_range)
y_pred_radio = model_radio.predict(budget_range)

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(budget_range, y_pred_tv, color='blue', label='Budget Télévision')
plt.plot(budget_range, y_pred_social, color='green', label='Budget Réseaux Sociaux')
plt.plot(budget_range, y_pred_radio, color='orange', label='Budget Radio')
plt.title('Régression Linéaire pour le Mix Média')
plt.xlabel('Budget')
plt.ylabel('Ventes')
plt.legend()
plt.grid(True)
plt.show()

