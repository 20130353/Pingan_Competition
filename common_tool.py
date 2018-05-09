import pandas as pd
import random
def write_result(id,pre_lable):
    """

    :param id:type-series
    :param pre_lable:type-array
    :return:nothing
    """
    dataframe = pd.DataFrame({'Id': id, 'Pred': pre_lable}, dtype=float)
    dataframe = pd.pivot_table(dataframe, index=['Id'])
    dataframe.to_csv("model/test.csv", index=True, sep=',')

def evaluate_feature(feature_name,data):
    index = list(data.columns).index(feature_name)
    train_data = data.drop(feature_name,axis=1)
    from sklearn.linear_model import LinearRegression
    regr = LinearRegression().fit(train_data,data[feature_name].values)
    # print(feature_name + '_cor:' + str(regr.coef_[index]))

def create_rand(c):
    global seed
    return random.uniform(0,seed)

def process_y0(data):
    data0 = data[data.Pred==0]
    global seed
    seed = data[data.Pred!=0].min()
    y = data0[['Pred']].apply(create_rand, axis=1)
    data0.Pred = y
    new_data = pd.concat([data0,data[data.Pred!=0]])

    return new_data