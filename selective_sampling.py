# Implementation of selective sampling
# delta: in (0, 1], confidence level
# tau: >=0, tolerance hyperparameter
# alpha: [0.01, 10] hyperparameter

# t: index of instance x
# j: index of teacher j
# K: number of teachers
# y: label for x
# theta: threshold
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable


tau = 0.3
alpha = 0.1

train_file = "./data/my_train.txt"
test_file = "./data/my_test.txt"


def preprocess(data_file):
    data = np.loadtxt(data_file)
    print('preprocess data:', data.shape)
    X = data[:, 2:]
    y = data[:, 0].astype(int)
    q_ids = data[:, 1].astype(int)
    print('X ', X.shape, 'y ', y.shape, 'q ids ', q_ids.shape)
    # print(X[0], y[0], q_ids[0])
    # normalize
    for i in range(X.shape[0]):
        X[i] = X[i] / np.linalg.norm(X[i])
    # print(X[0])
    return X, y, q_ids
    # SelectiveLearner(X_train)


class teacherClassifier(object):

    def __init__(self, X, y, q_ids, K=1):
        self.teacher = []
        self.loss_hist = {}
        q_type = np.unique(q_ids)
        q_len = q_type.shape[0]
        assign_queue = random.sample(list(q_type), q_len)
        partition = q_len // K
        for j in range(K):
            start = partition*j
            end = start + partition
            query_ids = assign_queue[start:end]
            query_index = np.where(np.isin(q_ids, query_ids))
            y_train = y[query_index]
            X_train = X[query_index]
            print("teacher:", j, "train size:", X_train.shape, "y size:", y_train.shape)
            teacher_model, loss_history = self.teacher_train(X_train, y_train)
            self.teacher.append(teacher_model)
            self.loss_hist['teacher'+str(j)] = loss_history
        self.loss_hist_display()
        print("teacher training completed:", len(self.teacher))

    def teacher_train(self, X, y):
        # train teacher classifier
        # N - num of samples
        N, input_dim = X.shape
        h1_dim, h2_dim, output_dim = input_dim, input_dim, 1
        h1_layer = nn.Linear(h1_dim, h2_dim)
        #torch.nn.init.kaiming_uniform(h1_layer.weight, mode='fan_in')
        h2_layer = nn.Linear(h2_dim, output_dim)
        #torch.nn.init.kaiming_uniform(h2_layer.weight, mode='fan_in')

        model = nn.Sequential(nn.Linear(input_dim, h1_dim),
                              nn.ReLU(),
                              h1_layer,
                              nn.ReLU(),
                              h2_layer)
        loss_fn = torch.nn.MSELoss(size_average=True)
        learning_rate = 1e-2
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # X = torch.from_numpy(X)
        X = Variable(torch.FloatTensor(X))
        y = y.tolist()
        y = Variable(torch.Tensor(y), requires_grad=False)
        loss_history = []

        for t in range(20000):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            if t % 500 == 0:
                loss_history.append(loss.data[0])
            # print(t, loss.data[0], y_pred[0], y[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model, loss_history
    
    def loss_hist_display(self):
        plt.title('Training loss')
        for teacher, loss_history in self.loss_hist.items():
            plt.plot(loss_history, '-o', label=teacher)
        plt.xlabel('Epoch')
        plt.savefig('./teacher_train_iter20000.png')
        #plt.show()

    def predict(self, j, x):
        x = Variable(torch.FloatTensor(x))
        y_pred = self.teacher[j](x)
        y_pred = y_pred.data[0]

        #import pdb;pdb.set_trace();
        if y_pred > 0:
            label = 1
        else:
            label = -1
        return label


class SelectiveLearner(object):

    def __init__(self, X, y, teachers, K=5):
        self.K = K  # number of teachers
        # self.teacher_knows = teacher_knows
        # print("shape of teacher knows:", teacher_knows.shape)
        num_samples, num_features = (X.shape[0], X.shape[1])
        self.X = np.insert(X, 0, np.zeros((num_features,)), axis=0)
        self.y = np.insert(y, 0, 0, axis=0)
        self.teachers = teachers
        # self.teacher_knows = np.insert(teacher_knows, 0, 0, axis=1)
        # print("shape of inserted X:", self.X.shape)
        self.w = np.zeros((K, 2, num_features))
        self.A = np.zeros((K, 2, num_features, num_features))
        for j in range(K):
            self.A[j][0] = np.eye(num_features)
        self.z_count = 0
        self.y_preds = [0]
        self.correctness = 1
        # import pdb;pdb.set_trace()

    def query(self, t, j):
        label = self.y[t]
        #label = self.teachers.predict(j, self.X[t])
        return label

    def predict(self, delta_hat):
        return np.sign(delta_hat)

    def selective_sampler(self, t):
        # dealing with t_th instance x_t
        if t % 5000 == 0:
            print("t:", t)
        theta_square = np.zeros(self.K)
        # print('theta_square:', theta_square)
        delta = np.zeros(self.K)
        for j in range(self.K):
            # pinv do SVD
            theta_square[j] = alpha * np.dot(self.X[t].T, np.linalg.pinv(self.A[j, 0])).dot(self.X[t]) * np.log(1+t)
            # print('theta_square:', j, theta_square[j])
            delta[j] = np.dot(self.w[j, 0].T, self.X[t])
        # print('theta_square:', theta_square)
        theta = np.sqrt(theta_square)
        j_t = np.argmax(np.absolute(delta))
        #print('j_hat[t]:', j_t)
        C = []
        H = []
        c_bound_base = np.absolute(delta[j_t]) - tau - theta[j_t]
        for j in range(self.K):
            c_bound = c_bound_base - theta[j]
            if np.absolute(delta[j]) >= c_bound:
                C.append(j)
        # Is it possible for C to be None?
        #print("size of confidence set C:", len(C))
        if len(C):
            h_bound_base = np.absolute(delta[j_t]) - tau + np.amax(theta[C])
            for i in C:
                h_bound = h_bound_base + theta[i]
                if np.absolute(delta[i]) >= h_bound:
                    H.append(i)
        B = np.array(list(set(C) - set(H)))
        delta_t = np.average(delta[C])
        # why need to predict?
        y_hat = self.predict(delta_t)
        if y_hat == self.y[t]:
            self.correctness += 1
        #self.y_preds.append(y_hat)
        # get value of Z
        Z = 0
        #print('B', B)
        len_B = B.shape[0]
        for i in range(2**len_B):
            e = list(bin(i))[2:]
            e = np.array(e) == '1'
            # print('e:', e)
            if len_B == 0:
                S = []
            else:
                S = B[len_B-len(e):][e]
            SH = list(S).extend(list(H))
            delta_sh = np.average(delta[SH])
            theta_sh = np.average(theta[SH])
            if delta_t * delta_sh <= 0 or np.absolute(delta_sh) < theta_sh:
                Z = 1
                break
        self.z_count += Z
        if Z == 1:
            for j in C:
                y = self.query(t, j)  # query y[j,t]
                self.A[j, 1] = self.A[j, 0] + self.X[t] * self.X[t].T
                tmp_w = self.w[j, 0]
                if np.absolute(delta[j]) > 1:
                    inv_A = np.linalg.pinv(self.A[j, 0])
                    tmp_w -= np.sign(delta[j]) * ((np.absolute(delta[j])-1)/np.dot(self.X[t].T.dot(inv_A), self.X[t])) * inv_A.dot(self.X[t])
                self.w[j, 1] = np.dot(np.linalg.pinv(self.A[j, 1]), (self.A[j, 0].dot(tmp_w) + y*self.X[t]))
        else:
            self.A[:, 1] = self.A[:, 0]
            self.w[:, 1] = self.w[:, 0]
        self.A[:, 0] = self.A[:, 1]
        self.w[:, 0] = self.w[:, 1]
        # print(self.A[j,t])

    def train(self):
        for t in range(1, self.X.shape[0]):
            self.selective_sampler(t)
        print("times of teacher queries:", self.z_count)
    
    def precision(self):
        y = self.y.tolist()
        total = len(y)
        print('sampler correctness:', self.correctness/total)


if __name__ == '__main__':
    # file_process(train_file)
    print("train data preprocessing...")
    X_train, y_train, q_ids_train = preprocess(train_file)
    unique, counts = np.unique(y_train, return_counts=True)
    print("y train:", np.asarray((unique, counts)).T)
    unique, counts = np.unique(q_ids_train, return_counts=True)
    q_ids_params = np.asarray((unique, counts)).T
    print("q ids types:", q_ids_params.shape[0], 'max:', q_ids_params[:, 1].max(), 'min:', q_ids_params[:, 1].min())

    print("test data preprocessing...")
    X_test, y_test, q_ids_test = preprocess(test_file)
    unique, counts = np.unique(y_test, return_counts=True)
    print("y test:", np.asarray((unique, counts)).T)
    unique, counts = np.unique(q_ids_test, return_counts=True)
    q_ids_params = np.asarray((unique, counts)).T
    print("q ids types:", q_ids_params.shape[0], 'max:', q_ids_params[:, 1].max(), 'min:', q_ids_params[:, 1].min())
    
    # train teacher classifiers
    K = 5
    #teachers = teacherClassifier(X_train, y_train, q_ids_train, K)
    teachers = []
    # sampling
    learner = SelectiveLearner(X_test, y_test, teachers, K)
    learner.train()
    learner.precision()





