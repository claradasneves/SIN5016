import numpy as np
from ..utils.activations import softmax
from ..utils.regularizations import lasso

class RegLog():
    """ regressão logística"""
    def __init__(
            self,
            num_features,
            num_classes,
            cost_function,
            regularization=None,
        ):
        self.M = num_features
        self.K = num_classes

        W = np.random.randn(self.M, self.K)
        self.W = np.insert(W, 0, 1, axis=0) # add bias

        self.cost_function = cost_function
        self.regularization=regularization

    def gradient_descent(
            self,
            X, y,
            alpha=0.001,
            batch_size=16,
        ):
        """"
        Args:
            x: input vector
            steps: qtdade de passos para andar na direção do gradiente
            alpha: escalar que reduz o tamanho do vetor gradiente, isso serve para eu saber a direção, mas andar passos curtos na direção indicada
            tol: o tamanho mínimo que eu espero que meu gradiente chegue. Isso indica o tamanho de tolerância para atualizações do gradiente.
        
        Returns
            o ponto mínimo local, no contexto dos hiperparâmetros.
        """

        N = X.shape[0]
        losses = []

        for start in range(0, N, batch_size):

            batch_x = X[start:start+batch_size]
            batch_y = y[start:start+batch_size]
            
            y_pred = self.predict(batch_x)

            loss = self.cost_function(batch_y, y_pred)
            losses.append(loss)

            # shape: (N, m).T -> (m, N) x (N, k) -> (m, k)
            dEdW = batch_x.T.dot(y_pred - batch_y) / batch_size

            # TODO regularization
            
            self.W -= alpha * dEdW
        
        losses = np.array(losses)

        return losses, dEdW
    
    def predict(self, X: np.array):
        """Modelo da Regressão Logística multinomial"""
        
        logit = X.dot(self.W)

        return softmax(logit=logit)
    
    def fit(
        self,
        X_train, y_train, X_test, y_test,
        optimizer='GD',
        epochs=1000, 
        learning_rate=0.001,
        tol=0.001,    
        ):

        # Prepara dados de treino e testa com a adição do intercepto
        X_train = np.insert(X_train, 0, 1, axis=1) # bias
        X_test = np.insert(X_test, 0, 1, axis=1)

        history_loss = []
        for epoch in range(epochs):

            if optimizer == 'GD':
                loss, dEdW = self.gradient_descent(
                    X=X_train, 
                    y=y_train,
                    alpha=learning_rate,
                )

            # Avaliação Final
            y_pred_val = self.predict(X_test)

            val_loss = self.cost_function(y_test, y_pred_val)
            history_loss.append([np.mean(loss), val_loss])

            if np.linalg.norm(dEdW) < tol: break

            print(f'epoch {epoch}: Avg training loss (GD)={np.mean(loss)} | val loss {val_loss}')

        history_loss = np.array(history_loss)

        return history_loss
    