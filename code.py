#Library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from google.colab import files

#Read file
df = pd.read_csv('TRANSACTIONS.csv')
df.head()

#Data preparation
##Pivot table
df['transaction_id'] = df['MemberID'].astype(str)+'_'+df['Order'].astype(str)
df_transform=pd.crosstab(df['transaction_id'], df['Item'])
#df_transform.head()

##One-hot encoding
def encode(item_freq):
    res = False
    if item_freq > 0:
        res = True
    return res

basket_input = df_transform.map(encode)
basket_input.head()

#Build the Apriori Algorithm
frequent_itemsets = apriori(basket_input, min_support=0.07, use_colnames=True, max_len =3)
rules = association_rules(frequent_itemsets,metric="lift",min_threshold=1,num_itemsets=len(basket_input))

#Filter the necessary data and save it according to the local computer
rules.sort_values(by=["lift", "confidence", "support"],axis = 0, ascending = [False, False, False]).head(20).to_csv('sorted_rules.csv', index=False)
files.download('sorted_rules.csv')

#Extra
##Visualize item frequency
df_frequency = df_transform.loc[:, ~df_transform.columns.isin(['transaction_id'])]
item_frequency=df_frequency.sum().sort_values(ascending=False)
color = plt.cm.rainbow(np.linspace(0, 1, 40))
item_frequency.head(40).plot.bar(color = color, figsize=(13,5))
plt.title('Frequency of most popular items', fontsize = 20)
plt.xticks(rotation = 90 )
plt.grid()
plt.show()
