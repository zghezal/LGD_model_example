import pandas as pd
import numpy as np
import math
from scipy.stats import shapiro, pearsonr
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

# 1️⃣ Chargement des données
df = pd.read_csv('Dataset_LGD_Synthetique.csv')

# 2️⃣ Enrichissement et Traitement de la data en général
# On ne modélise que sur les défauts
df = df[df['default_flag'] == 1].copy()

# 3️⃣ Calcul de la LGD : Variable à modéliser
# LGD
df['lgd'] = np.where(
    df['default_flag'] == 1,
    1 - df['recovered_amount'] / df['exposure_at_default'],
    0
)
# High_loss (equivaut à la variable à modéliser qualitative)
df['high_loss'] = (df['lgd'] > 0.5).astype(int)

# 3️⃣ Étape 2 : Validation des variables explicatives
numerical_vars = ['loan_age', 'collateral_value', 'revenue']
outlier_counts = {}
normality_pvals = {}

for var in numerical_vars:
    z_scores = (df[var] - df[var].mean()) / df[var].std()
    outlier_counts[var] = (np.abs(z_scores) > 3).sum()
    _, pval = shapiro(df[var])
    normality_pvals[var] = pval

print("🔍 Outliers détectés par variable :", outlier_counts)
print("Utilité : détecter les observations extrêmes ou erreurs de saisie pour décider de les exclure ou de les traiter.\n")

print("📊 P-values du test de normalité (Shapiro-Wilk) :", normality_pvals)
print("Utilité : évaluer si les variables numériques sont normalement distribuées, afin de choisir des transformations ou tests statistiques appropriés.\n")

# 4️⃣ Étape 3 : Tests d’explicativité (corrélations)
corrs = {}
# Numériques
for var in numerical_vars:
    corrs[var] = pearsonr(df[var], df['lgd'])[0]
# Catégorielles
categorical_vars = [
    'loan_type', 'rating', 'sector',
    'country', 'deal_type', 'revolving', 'covenant'
]
dummies = pd.get_dummies(df[categorical_vars], drop_first=True)
for col in dummies.columns:
    corrs[col] = pearsonr(dummies[col], df['lgd'])[0]

print("🔗 Corrélations entre variables explicatives et LGD :", corrs)
print("Utilité : mesurer la force et le signe de la relation pour sélectionner les variables les plus explicatives.\n")

# 5️⃣ Étape 4 : Modélisation logistique
features = numerical_vars + categorical_vars
X = df[features]
y = df['high_loss']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_vars),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_vars)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
ll = log_loss(y_test, y_pred_proba)

print(f"📈 AUC ROC : {auc:.3f}")
print("Utilité : évaluer la capacité du modèle à discriminer les fortes pertes (haut vs bas risque).\n")

print(f"📉 Log-loss : {ll:.3f}")
print("Utilité : mesurer la qualité des probabilités prédites par le modèle (plus bas = meilleur).")

numerical_vars = ['loan_age','collateral_value','revenue']
categorical_vars = [
    'loan_type','rating','sector',
    'country','deal_type','revolving','covenant'
]
all_features = numerical_vars + categorical_vars
# Split train/test
X = df[all_features]
y = df['high_loss']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

def evaluate(features):
    # Préparation du préprocesseur
    transformers = []
    num = [f for f in features if f in numerical_vars]
    cat = [f for f in features if f in categorical_vars]
    if num:
        transformers.append(('num', StandardScaler(), num))
    if cat:
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False), cat))
    preprocessor = ColumnTransformer(transformers, remainder='drop')

    # Pipeline avec logistic regression
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Entraînement
    pipeline.fit(X_train[features], y_train)

    # Prédictions sur test
    proba_test = pipeline.predict_proba(X_test[features])[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    ll = log_loss(y_test, proba_test)

    # Calcul du BIC sur l'échantillon d'entraînement
    proba_train = pipeline.predict_proba(X_train[features])[:, 1]
    # Éviter log(0)
    eps = 1e-15
    p = np.clip(proba_train, eps, 1 - eps)
    loglik = np.sum(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))
    n = len(y_train)
    k = pipeline.named_steps['clf'].coef_.shape[1] + 1  # paramètres + intercept
    bic = -2 * loglik + k * math.log(n)

    return auc, ll, bic

# 4. Évaluation des modèles
results = {}
results['global'] = evaluate(all_features)
for feat in all_features:
    results[feat] = evaluate([feat])

# 5. Affichage comparatif
print(f"{'Feature':<20} {'AUC':<8} {'LogLoss':<10} {'BIC'}")
print("-" * 60)
for feat, (auc, ll, bic) in results.items():
    name = 'global (toutes)' if feat == 'global' else feat
    print(f"{name:<20} {auc:.3f}   {ll:.3f}   {bic:.1f}")