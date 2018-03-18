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


tau = 0.3
alpha = 0.1
input_shape = (130, 1)

train_file = "./data/toy.txt"


def preprocess(train_file):
    train_data = np.loadtxt(train_file)
    print('train data:', train_data.shape)
    X_train = train_data[:, 2:]
    y_train = train_data[:, 0].astype(int)
    q_ids = train_data[:, 1].astype(int)
    print('X ', X_train.shape, 'y ', y_train.shape, 'q ids ', q_ids.shape)
    # print(X_train[0], y_train[0], q_ids[0])
    # normalize
    for i in range(X_train.shape[0]):
        X_train[i] = X_train[i] / np.linalg.norm(X_train[i])
    # print(X_train[0])
    return X_train, y_train, q_ids
    # SelectiveLearner(X_train)


def teacher_train(y, q_ids, K):
    teacher_knows = np.zeros([K, y.shape[0]])
    q_type = np.unique(q_ids)
    q_len = q_type.shape[0]
    assign_queue = random.sample(list(q_type), q_len)
    partition = q_len // 5
    for j in range(K):
        start = partition*j
        end = start + partition
        query_ids = assign_queue[start:end]
        query_index = np.where(np.isin(q_ids, query_ids))[0]
        teacher_knows[j][query_index] = y[query_index]

    return teacher_knows


class SelectiveLearner(object):

    def __init__(self, X, teacher_knows, K=5):
        self.K = K  # number of teachers
        # self.teacher_knows = teacher_knows
        # print("shape of teacher knows:", teacher_knows.shape)
        num_samples, num_features = (X.shape[0], X.shape[1])
        self.X = np.insert(X, 0, np.zeros((num_features,)), axis=0)
        self.teacher_knows = np.insert(teacher_knows, 0, 0, axis=1)
        # print("shape of new teacher knows:", self.teacher_knows.shape)
        # print("shape of inserted X:", self.X.shape)
        self.w = np.zeros((K, num_samples+1, num_features))
        self.A = np.zeros((K, num_samples+1, num_features, num_features))
        for j in range(K):
            self.A[j][0] = np.eye(num_features)
        self.z_count = 0
        # import pdb;pdb.set_trace()

    def query(self, t, j):
        label = self.teacher_knows[j, t]
        if label == 0:
            label = random.choice([-1, 1])
        return label

    def predict(self, delta_hat):
        return np.sign(delta_hat)

    def selective_sampler(self, t):
        # dealing with t_th instance x_t
        print("t:", t) 
        theta_square = np.zeros(self.K)
        # print('theta_square:', theta_square)
        delta = np.zeros(self.K)
        for j in range(self.K):
            # pinv do SVD
            theta_square[j] = alpha * np.dot(self.X[t].T, np.linalg.pinv(self.A[j, t-1])).dot(self.X[t]) * np.log(1+t)
            # print('theta_square:', j, theta_square[j])
            delta[j] = np.dot(self.w[j, t-1].T, self.X[t])
        # print('theta_square:', theta_square)
        theta = np.sqrt(theta_square)
        j_t = np.argmax(np.absolute(delta))
        print('j_hat[t]:', j_t)
        C = []
        H = []
        c_bound_base = np.absolute(delta[j_t]) - tau - theta[j_t]
        for j in range(self.K):
            c_bound = c_bound_base - theta[j]
            if np.absolute(delta[j]) >= c_bound:
                C.append(j)
        # Is it possible for C to be None?
        print("size of confidence set C:", len(C))
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
        # get value of Z
        Z = 0
        print('B', B)
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
                self.A[j, t] = self.A[j, t-1] + self.X[t] * self.X[t].T
                tmp_w = self.w[j, t-1]
                if np.absolute(delta[j]) > 1:
                    inv_A = np.linalg.pinv(self.A[j, t-1])
                    tmp_w -= np.sign(delta[j]) * ((np.absolute(delta[j])-1)/np.dot(self.X[t].T.dot(inv_A), self.X[t])) * inv_A.dot(self.X[t])
                self.w[j, t] = np.dot(np.linalg.pinv(self.A[j, t]), (self.A[j, t-1].dot(tmp_w) + y*self.X[t]))
        else:
            self.A[j, t] = self.A[j, t-1]
            self.w[j, t] = self.w[j, t-1]
        # print(self.A[j,t])

    def train(self):
        for t in range(1, self.X.shape[0]):
            self.selective_sampler(t)
        print(self.z_count)


if __name__ == '__main__':
    # file_process(train_file)
    X_train, y_train, q_ids = preprocess(train_file)
    unique, counts = np.unique(y_train, return_counts=True)
    print("y train:", np.asarray((unique, counts)).T)
    unique, counts = np.unique(q_ids, return_counts=True)
    q_ids_params = np.asarray((unique, counts)).T
    print("q ids types:", q_ids_params.shape[0], 'max:', q_ids_params[:, 1].max(), 'min:', q_ids_params[:, 1].min())
    K = 5
    teacher_knows = teacher_train(y_train, q_ids, K)
    learner = SelectiveLearner(X_train, teacher_knows, K)
    learner.train()




