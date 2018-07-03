import pandas as pd
import numpy as np
x =[[1,2,3],[3,4,5],[3,4,5]]
df = pd.DataFrame(x,columns=['a','b','c'])
def fun(x):
    x.info()
    print(x.columns)
    print(x)

print(df.groupby(by=['a','b']))
print('fasdfadsdfa')
# data = df.groupby(by=['a','b']).apply(fun(x))
# data = df.groupby(by=['a','b']).transform(fun(x))
data = df.groupby(by=['a','b']).agg(fun(x))
print(data)