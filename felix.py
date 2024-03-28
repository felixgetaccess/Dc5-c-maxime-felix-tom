import pandas as pd
from sklearn.linear_model import LinearRegression

# Chargement des données
sales_data = pd.read_csv('./supermarket_sales.csv')
marketing_data = pd.read_excel('./dataset_marketing_grand.xlsx')

# Préparation des données et sélection des colonnes pour les budgets et les ventes
# Note : Remplacez 'NomColonne' par les noms de colonne appropriés
X_tv = marketing_data['Budget Télévision'].values.reshape(-1, 1)
X_social = marketing_data['Budget Réseaux Sociaux'].values.reshape(-1, 1)
y_sales = marketing_data['Total'].values

# Entraînement des modèles de régression
model_tv = LinearRegression().fit(X_tv, y_sales)
model_social = LinearRegression().fit(X_social, y_sales)

