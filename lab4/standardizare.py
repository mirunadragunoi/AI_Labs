from sklearn import preprocessing
import numpy as np

x_train = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]], dtype=np.float64)
x_test = np.array([[-1, 1, 0]], dtype=np.float64)

# facem statisticile pe datele de antrenare
scaler = preprocessing.StandardScaler()
scaler.fit(x_train)

# afisam media
print(scaler.mean_) # => [1. 0. 0.33333333]

# afisam deviatia standard
print(scaler.scale_) # => [0.81649658 0.81649658 1.24721913]

# scalam datele de antrenare
scaled_x_train = scaler.transform(x_train)
print(scaled_x_train) # => [[0. -1.22474487 1.33630621]

# [1.22474487 0. -0.26726124]
# [-1.22474487 1.22474487 -1.06904497]]
# scalam datele de test
scaled_x_test = scaler.transform(x_test)
print(scaled_x_test) # => [[-2.44948974 1.22474487 -0.26726124]]