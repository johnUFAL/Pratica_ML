#hora de escolher um k bom e pelo gráfico é 2
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(dados_normalizados)

#rotulos ao original
df['Cluster'] = clusters

#cluster com Scatter Plot 2D: Al vs k_avg
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='AL', y='k_avg', hue='Cluster', palette='rainbow', s=70)
plt.title('AL vs k_avg')
plt.grid(True)
plt.show()

#cluster com Scatter Plot 2D:  WTW vs ACD
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='WTW', y='ACD', hue='Cluster', palette='rainbow', s=70)
plt.title('WTW vs ACD')
plt.grid(True)
plt.show()


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