import numpy as np
import pandas as pd

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, 0: 3].values  # iloc[row, column]
Y = dataset.iloc[:, 3].values


# use class Imputer from sklearn.preprocessing to handle the missing Data
from sklearn.preprocessing import Imputer

"""
sklearn.preprocessing.Imputer(missing_values=’NaN’, strategy=’mean’, axis=0, verbose=0, copy=True)

missing_values：缺失值，可以为整数或NaN(缺失值numpy.nan用字符串‘NaN’表示)，默认为NaN

strategy：替换策略，字符串，默认用均值‘mean’替换
①若为mean时，用特征列的均值替换
②若为median时，用特征列的中位数替换
③若为most_frequent时，用特征列的众数替换

axis：指定轴数，默认axis=0代表列，axis=1代表行

copy：设置为True代表不在原数据集上修改，设置为False时，就地修改，存在如下情况时，即使设置为False时，也不会就地修改
"""

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1: 3])
X[:, 1: 3] = imputer.transform(X[:, 1: 3])


# use class LabelEncode from sklearn.preprocessing to Encode categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


# create a dummy variable
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# use class train_test_split from sklearn.preprocessing to splite the datasets into trainsets and test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)



# use class StandardScalar from sklearn.preprocessing to feature scale
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

print(X_train)
print(X_test)