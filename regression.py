import numpy as np

class LinearRegression:
    def __init__(self):
        self.w =  None

    def pred(self, X, w):
        return np.dot(X, w)
    
    def _prepare(self, X, y):
        # estandarizar X
        X = np.asarray(X).astype(float)
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0, ddof=0)
        x_std[x_std == 0.0] = 1.0  # evitar división por cero
        Xs = (X - x_mean) / x_std

        # estandarizar y
        y = np.asarray(y).reshape(-1, 1).astype(float)
        y_mean = np.mean(y)
        y_std = np.std(y, axis=0, ddof=0)
        if y_std == 0.0:
            y_std = 1.0
        ys = (y - y_mean) / y_std

        Xb = self._add_bias(Xs)
        return Xb, ys

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _loss(self, Xb, y_real, lambda_):
        n = Xb.shape[0]
        y_pred = self.pred(Xb, self.w)
        mse = np.mean((y_real - y_pred) ** 2)
        w_nb = self.w[1:]  # no regularizar el bias
        reg = (lambda_ / n) * np.sum(w_nb ** 2)
        return mse + reg

    def _gradient(self, Xb, y_real, lambda_):
        n = Xb.shape[0]
        y_pred = self.pred(Xb, self.w)
        grad = (-2.0 / n) * np.dot(Xb.T, y_real - y_pred)
        grad_reg = np.zeros_like(self.w)
        grad_reg[1:] = (2 * lambda_ / n) * self.w[1:]  # no regularizar el bias
        return grad + grad_reg

    # lambda_: regularizador (qué tanto penaliza la complejidad del modelo)
    def fit(self, X, y, alpha, lambda_, n_iter):
        X_real, y_real = self._prepare(X, y)
        self.w = np.zeros((X_real.shape[1], 1), dtype=float)  # initialize weights

        loss_list, weight_list = [], []
        for i in range(n_iter):
            # gradient descendent step
            grad = self._gradient(X_real, y_real, lambda_)
            self.w -= alpha * grad
            weight_list.append(self.w.copy())
            # get loss
            loss = self._loss(X_real, y_real, lambda_)
            loss_list.append(float(loss))
            # porcentaje de 20 en 20
            if (i + 1) % (n_iter // 20) == 0:
                print(f"Porcentaje {i / n_iter * 100:.0f}%, Loss: {loss}")
        
        return loss_list, weight_list