import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
x = df[cols_espessura]
escala = StandardScaler()
dados_normalizados = escala.fit_transform(x)
df[cols_espessura] = dados_normalizados

#metodo cotovelo para k
inercia = []
k_raio = range(1, 31)

for k in k_raio:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(dados_normalizados)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_raio, inercia, marker='o')
plt.xlabel('Num Clusters (k)')
plt.ylabel('Inercia')
plt.title('Cotovelo')
plt.grid(True)
plt.show()
