import pandas as pd
import numpy as np
x =[[1,2.01010101010010,3],[3,4,5],[3,4,5]]
df = pd.DataFrame(x,columns=['a','b','c'])
df.loc[1,'a']=1000
df.loc[1:2,'b'] = 2000
df = df.loc[:,['a','b']].astype('int8')
df.info(memory_usage='deep')
print(df)

print(np.zeros(df.shape[0]))