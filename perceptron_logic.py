import numpy as np

class PerceptronSimple:
    def __init__(self):
        self.weights = None
        self.threshold = 0
        self.lr = 0.1
        self.max_iter = 100
        self.max_error = 0.01
        self.error_history = []
        self.trained = False

    def step(self, x):
        return 1 if x >= 0 else 0

    def initialize(self, n_features):
        self.weights = np.random.uniform(-1, 1, n_features)
        self.threshold = np.random.uniform(-1, 1)
        self.error_history = []
        self.trained = False

    def train(self, X, y, callback=None):
        self.error_history.clear()
        for epoch in range(self.max_iter):
            total_err = 0
            for xi, target in zip(X, y):
                net = np.dot(xi, self.weights) - self.threshold
                out = self.step(net)
                error = target - out
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.threshold -= self.lr * error
                total_err += abs(error)
            avg = total_err / len(y)
            self.error_history.append(avg)
            if callback:
                callback(epoch+1, avg)
            if avg <= self.max_error:
                self.trained = True
                return f"✅ Entrenado en {epoch+1} iteraciones"
        self.trained = True
        return f"⚠️ Alcanzado máx iteraciones"

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self.step(np.dot(xi, self.weights)-self.threshold) for xi in X])
