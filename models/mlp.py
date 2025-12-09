import numpy as np

class MLP():
    """multi-layer perceptron"""
    def __init__(
            self,
            num_features,
            num_classes,
            num_neurons,
            hidden_layer_activation,
            output_layer_activation,
        ):       
        self.M = num_features
        self.H = num_neurons # qtdade de neurônios na camada escondida
        self.K = num_classes # qtdade de classes para predição

        # W: matriz com pesos da camada de entrada
        W = np.random.randn(self.M, self.H) * 0.01 # shape: (M, H)
        self.W = np.insert(W, 0, 1, axis=0) # add bias | shape: (M+1, H)
        
        # V: matriz com pesos da camada de saida
        self.V = np.random.randn(self.H, self.K) * 0.01 # shape: (H, K)

        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation

    def gradient_descent(
            self,
            cost_function,
            X, y,
            alpha=0.001,
            batch_size=16,
        ):
                    
        N = X.shape[0]
        losses = []

        for idx in range(0, N, batch_size):

            batch_x = X[idx:idx+batch_size]
            batch_y = y[idx:idx+batch_size]
        
            y_pred = self.predict(batch_x) # shape: (N, k)
            
            logit = batch_x.dot(self.W) # shape: (N, m) x (m, h) -> (N, h)
            w_hidden = self.hidden_layer_activation(logit) # shape: (N, h)
            
            loss = cost_function(batch_y, y_pred) # shape: (N, k)
            losses.append(loss)

            grad_erro = (batch_y - y_pred) # shape: (N, k)
            
            erro_propagado = grad_erro.dot(self.V.T) # shape: (N, k) x (k, h) -> (N, h)
            
            Delta1 = \
                erro_propagado * self.hidden_layer_activation(
                    logit, derivative=True,
                ) # shape: (N, h) * (N, h) -> (N, h)
            
            dEdV = grad_erro.T.dot(w_hidden) # shape: (k, N) x (N, h) -> (k, h)
            dEdW = batch_x.T.dot(Delta1) / batch_size # shape: (m, N) x (N, h) -> (m, h)
            
            self.W -= alpha * dEdW # shape: (m, h) * (m, h)
            self.V -= alpha * dEdV.T # shape: (h, k) * (h, k)
                    
        losses = np.array(losses)

        return losses, dEdW, dEdV

    def predict(self, X: np.array):
        """
        Função de predição do Perceptron multicamada

        Args
            X: matriz com os dados

        Returns
            lista de rótulos preditos
        """

        logit = X.dot(self.W) # shape: (N, m) x (m, h) -> (N, h)

        z_hidden = self.hidden_layer_activation(logit) # shape: (N, h)

        # isso pq estou em um problema multi-classe
        # se binário, eu usaria o hadamard-product
        z_hidden = z_hidden.dot(self.V) # shape: (N, h) x (h, k) -> (N, k)

        return self.output_layer_activation(z_hidden) # shape: (N, k)
    
    def fit(
            self,
            cost_function,
            X_train, y_train, X_test, y_test,
            optimizer='GD',
            epochs=1000, 
            learning_rate=0.001,
            tol=0.01,
        ):
        """
        Função de treino para a MLP
        """
        print('='*10)
        print(f'Training classification model with {optimizer}')

        # Prepara dados de treino e testa com a adição do intercepto
        X_train = np.insert(X_train, 0, 1, axis=1) # bias
        X_test = np.insert(X_test, 0, 1, axis=1)

        history_loss = []
        for epoch in range(epochs):
        
            if optimizer == 'GD':
                loss, dEdW, dEdV = self.gradient_descent(
                    cost_function=cost_function,
                    X=X_train, y=y_train,
                    alpha=learning_rate,
                )

            elif optimizer == 'newton':
                # TODO
                pass
            elif optimizer == 'bfgs':
                # TODO
                pass
            elif optimizer == 'polak-ribiere':
                # TODO
                pass
            elif optimizer == 'fletcher-reeves':
                # TODO
                pass

            # Avaliação Final
            y_pred_val = self.predict(X_test)

            val_loss = cost_function(y_test, y_pred_val)

            # Verifica a convergência baseado na norma dos gradientes
            if np.linalg.norm(dEdW) < tol and np.linalg.norm(dEdV) < tol: 
                print(f'Convergência atingida no Gradiente | Época {epoch}')
                break

            print(f'epoch {epoch}: Avg training loss (GD)={np.mean(loss)} | val loss {val_loss}')
            
            history_loss.append([np.mean(loss), val_loss])
        
        history_loss = np.array(history_loss)

        return history_loss