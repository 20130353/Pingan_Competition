import pandas as pd
x =[[1,2,3],[3,4,5],[3,4,5]]
df = pd.DataFrame(x,columns=['a','b','c'])
df.loc[1,'a']=1000
df.loc[1:2,'b'] = 2000

print(df)