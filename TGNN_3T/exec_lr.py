import numpy
import pandas
from sklearn.linear_model import LinearRegression


data = numpy.array(pandas.read_csv('../data/crystal/nlhm/id_target.csv'))
numpy.random.shuffle(data)
num_train = int(data.shape[0] * 0.8)

reg = LinearRegression().fit(data[:num_train, 1].reshape(-1, 1), data[:num_train, 2].reshape(-1, 1))
pred = reg.predict(data[num_train:, 1].reshape(-1, 1))

sum = numpy.abs(data[num_train:, 2].reshape(-1, 1) - pred)
print(numpy.mean(sum))
sum = (data[num_train:, 2].reshape(-1, 1) - pred)**2
print(numpy.sqrt(numpy.mean(sum)))
