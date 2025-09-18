import numpy as np

class PerceptronSimple:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = None
        self.threshold = None
        self.learning_rate = None
        self.error_history = []
        
    def initialize_parameters(self, weights=None, threshold=None, learning_rate=0.1):
        if weights is None:
            self.weights = np.random.rand(self.n_inputs) * 2 - 1  # Pesos entre -1 y 1
        else:
            self.weights = np.array(weights)
            
        if threshold is None:
            self.threshold = np.random.rand() * 2 - 1  # Umbral entre -1 y 1
        else:
            self.threshold = threshold
            
        self.learning_rate = learning_rate
    
    def activation_function(self, x):
        """Función de activación escalón"""
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        """Realiza una predicción"""
        summation = np.dot(inputs, self.weights) - self.threshold
        return self.activation_function(summation)
    
    def train(self, X, y, max_iterations=100, max_error=0.01):
        """Algoritmo de entrenamiento con Regla Delta"""
        self.error_history = []
        
        for iteration in range(max_iterations):
            total_error = 0
            
            for i in range(len(X)):
                # Predicción
                prediction = self.predict(X[i])
                error = y[i] - prediction
                
                # Actualización de pesos (Regla Delta)
                self.weights += self.learning_rate * error * X[i]
                self.threshold -= self.learning_rate * error
                
                total_error += abs(error)
            
            # Error promedio por patrón
            avg_error = total_error / len(X)
            self.error_history.append(avg_error)
            
            # Verificar condición de parada
            if avg_error <= max_error:
                return True, iteration + 1, avg_error
        
        return False, max_iterations, self.error_history[-1]