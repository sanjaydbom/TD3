import numpy as np

class OUP():
    def __init__(self, dims, mean = 0, theta = 0.15, sigma = 0.2):
        self.dims = dims
        self.mean = mean
        self.theta = theta
        self.sigma = sigma
        self.x_t = np.zeros(dims, dtype = np.float32)

    def reset(self):
        self.x_t = np.zeros(self.dims, dtype = np.float32)

    def __call__(self):
        dW_t = np.random.randn(self.dims)
        dx_t = self.theta * (self.mean - self.x_t) + self.sigma * dW_t
        self.x_t += dx_t
        return self.x_t