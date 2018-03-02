# Implementation of selective sampling
# delta: in (0, 1], confidence level
# tau: >=0, tolerance hyperparameter
# alpha: [0.01, 10] hyperparameter

# t: index of instance x
# j: index of teacher j
# K: number of teachers
# y: label for x
# theta: threshold


import math
import numpy as np

tau = 0.3
alpha = 0.1
num_inputs = 2000
input_shape = (1, 136)
y = np.array()  # shape [j, t] queried label of instance x_t from teacher j
y_hat = np.array()  # shape [t, 1] predicted answer from learner 


class SelectiveLearner(object):

    def init(self, K=5):
        self.K = K  # number of teachers

    def receive():
        x = np.array()
        return


    def query(self, x, t, j):
        # query label from teacher
        return


    A[j, 0] = np.identity()
    w[j, 0] = 0


    def predict(self, delta_hat, t):
        return np.sign(delta_hat[t])


    def selective_sampler(self, x, t):
        # dealing with t_th instance x_t
        x = receive()
        for j in range(K):
            theta_square[j, t] = alpha * x[t].T * np.linalg.inv(A[j,t-1]) * x[t] * np.log(1+t)
            delta_hat[j, t] = w[j, t-1].T * x[t]
        theta = np.sqrt(theta_square)
        j_hat[t] = np.argmax(np.absolute(delta_hat[:, t]), axis=1)
        print('j_hat[t]:', j_hat[t])
        C_hat[t] = []
        H_hat[t] = []
        for j in range(K):
            c_bound = np.absolute(delta_hat[j_hat[t], t]) - tau - theta[j, t] - theta[j_hat[t], t]
            if np.absolute(delta_hat[j, t]) >= c_bound:
                C_hat[t].append(j)
        max_theta = np.amax(theta[C_hat[t], t])
        for i in C_hat[t]:
            h_bound = np.absolute(delta_hat[j_hat[t], t]) - tau + theta[i, t] + max_theta
            if np.absolute(delta_hat[j, t]) >= max_theta:
                H_hat[t].append(i)
        B_hat[t] = C_hat[t] - H_hat[t]
        
        y[j] = query()
        y_hat[t] = self.predict(delat_hat, t)
        # TODO: not sure with Z's criteria here
        if True:
            Z[t] = 1
        else:
            Z[t] = 0

        if Z[t] == 1 and j in C_hat[t]:
            y[j, t] = query(x, t, j)  # query y[j,t]
            A[j, t] = A[j, t-1] + x[t] * x[t].T
            r[j,t] = x[t].T * np.linalg.inv(A[j,t]) * x[t]
            if np.absolute(delta_hat(j,t)) > 1:
                tmp_w[j, t-1] = w[j, t-1] - np.sign(delta_hat[j,t]) * ( (np.absolute(delta_hat(j,t))-1)/(x[t].T*np.linalg.inv(A[j,t-1])*x[t])) * np.linalg.inv(A[j,t-1]) * x[t]
            else:
                tmp_w = w[j, t-1]
            w[j,t] = np.linalg.inv(A[j,t]) * (A[j,t-1]*tmp_w + y[j,t]x[t])
        else:
            A[j, t] = A[j, t-1]
            r[j, t] = 0
            w[j, t] = w[j, t-1]

    def train(self, x):
        for t in range(num_inputs):
            self.selective_sampler(x, t)


if __name__ == '__main__':
    instances = None
    x = shape()
    learner = SelectiveLearner()
    for t in num_inputs:
        learner.train(x)


