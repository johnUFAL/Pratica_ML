import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#dados 'AL,	ACD,	WTW,	K1, 	K2'
path = '../RTVue_20221110.xlsx'
df = pd.read_excel(path)

df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
df['Eye'] = df['Eye'].map({'OS': 0, 'OD': 1})

sns.pairplot(df[['Age', 'Gender', 'C', 'S', 'ST', 'T', 
                 'IT', 'I', 'IN', 'N', 'SN']], height=1.8)
plt.show()
kind="hist"

var = ['Age', 'Gender', 'C', 'S', 'ST', 
       'T', 'IT', 'I', 'IN', 'N', 'SN']
df_select = df[var]
relacao = df_select.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(relacao, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlação")
plt.tight_layout()
plt.show()
