import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import silhouette_score

#dados 
path = 'RTVue_20221110.xlsx'
df = pd.read_excel(path)

#como ha dados faltantes, usar media, porem por genero
media_genero = df.groupby('Gender')[['C', 'S', 'ST', 'T', 'IT',
                                     'I', 'IN', 'N', 'SN']].mean()

for colunas in ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']:
    df[colunas] = df.apply(
        lambda row: media_genero.loc[row['Gender'], colunas]
        if pd.isnull(row[colunas]) else row[colunas],
        axis = 1
    )

#tratamento de dados categoricos
##label pra gender (M=1, F=0)
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
encoder = OneHotEncoder(sparse_output=False, drop='first')
eye_encoded = encoder.fit_transform(df[['Eye']])
df['Eye_OD'] = eye_encoded[:, 0] #OD=1, OS=0
df.drop('Eye', axis=1, inplace=True)

#normalizando os dados 
cols_espessura = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
x = df[cols_espessura].copy()
escala = StandardScaler()
dados_normalizados = escala.fit_transform(x)
df[cols_espessura] = dados_normalizados

#armazenar os scores
silhouette_scores = []

#testes de k ate 30
for k in range(2, 31):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(dados_normalizados)
    score = silhouette_score(dados_normalizados, labels)
    silhouette_scores.append(score)
    print(f"k={k}: Silhouette Score = {score:.3f}")

#plot
plt.figure(figsize=(8, 4))
plt.plot(range(2, 31), silhouette_scores, marker='o', color='teal')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score para Determinar k Ideal')
plt.grid(True)
plt.show()

