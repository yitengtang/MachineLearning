import numpy as np
from utils.features import prepare_for_training

class LinearRegression:
    def __init__(self, data, labels, polynominal_degree = 0, sinusoid_degree = 0,normalize_data=True):
    #1. initial all the parameters
    #2. get all the features
    #3. initial the theta array

        (data_processed,
         features_mean,
         features_deviation)=prepare_for_training(data,polynomial_degree = 0, sinusoid_degree = 0, normalize_data =True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynominal_degree = polynominal_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]#shape is hang he lie
        self.theta = np.zeros((num_features,1)) #init theta

    def train(self, alpha, num_iterations = 500):
        #train excute gradient descent
        cost_history = self.gradient_descent(alpha,num_iterations)
        return self.theta, cost_history


    def gradient_descent(self,alpha,num_iterations):
        #die dai mokuai
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history

    def gradient_step(self,alpha):
        #gradient reduce calculation method
        num_examples =self.data.shape[0] #num of samples
        prediction = LinearRegression.hyphothesis(self.data,self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        self.theta =theta

    def cost_function(self,data,labels):
        #loss function calculation
        num_examples = data.shape[0]#number of samples
        delta = LinearRegression.hyphothesis(self.data,self.theta) - labels
        cost =(1/2)*np.dot(delta.T,delta)/num_examples
        return cost[0][0]


    @staticmethod
    def hyphothesis(data,theta):
        predictions = np.dot(data,theta)
        return predictions

    def get_cost(self,data,labels):
        data_processed = prepare_for_training(data,
         self.polynominal_degree,
         self.sinusoid_degree,
         self.normalize_data
         ) [0]
        return self.cost_fuction(data_processed,labels)
    def predict(self,data):
        #
        data_processed = prepare_for_training(data,
                                            self.polynominal_degree,
                                            self.sinusoid_degree,
                                            self.normalize_data
                                            )[0]
        predictions = LinearRegression.hyphothesis(data_processed,self.theta)
        return predictions



