import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from LinearRegressionYiteng import LinearRegression

data =pd.read_csv('../data/world-happiness-report-2017.csv')

train_data = data.sample(frac = 0.8)

test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.' #x
output_param_name = 'Happiness.Score' #y

x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

plt.scatter(x_train, y_train, label ='Train data')
plt.scatter(x_test, y_test, label ='Test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.legend()
plt.title('Happy')
plt.show()

num_iterations = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train, y_train)
(theta,cost_history)=linear_regression.train(learning_rate, num_iterations)

print('lost from begining:', cost_history[0])
print('lost after train:', cost_history[-1])

plt.plot(range(num_iterations),cost_history)
plt.xlabel('Iter')
plt.ylabel('loss')
plt.title('GD')
plt.show()

predictions_num = 100
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train,y_train,label ='Train data')
plt.scatter(x_test,y_test,label ='test data')
plt.plot(x_predictions,y_predictions,'r',label ='prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Linear Regression')
plt.show()





