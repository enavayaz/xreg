import numpy as np
#from morphomatics.stats.exponential_barycenter import ExponentialBarycenter
#from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error

class Error:
    def __init__(self, dist, y, ytrue, n_learn=0):
        y, ytrue = y[n_learn:], ytrue[n_learn:]
        m = np.min([len(y), len(ytrue)])
        assert m != 0, "Empty List!"
        self.diff_dist = np.array([dist(y[i], ytrue[i]) for i in range(m)])
        self.dist = dist
        self.y, self.ytrue = y, ytrue

    def mae(self):
        return np.mean(self.diff_dist)

    def maxerr(self):
        return np.max(self.diff_dist)

    def mse(self):
        return np.square(self.diff_dist).mean()

    def r2(self, y_mean):
        ytrue, y = self.ytrue, self.y
        #return 1 - np.sum(self.dist(ytrue, y)**2)/np.sum(self.dist(ytrue, y_mean)**2)
        a = np.sum([self.dist(y[k], ytrue[k])**2 for k in range(len(ytrue))])
        b = np.sum([self.dist(y_mean, ytrue[k])**2 for k in range(len(ytrue))])
        return 1-a/b
def errfun(dist):
    return lambda y, ytrue, n_learn=0: Error(dist, y, ytrue, n_learn)
