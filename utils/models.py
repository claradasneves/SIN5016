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
        W = np.random.randn(self.M, self.H) * 0.001 # shape: (M, H)
        self.W = np.insert(W, 0, 1, axis=1) # add bias
        
        # V: matriz com pesos da camada de saida
        V = np.random.randn(self.H, self.K) * 0.001 # shape: (H, K)
        self.V = np.insert(V, 0, 1, axis=1) # add bias

        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation

    def gradient_descent(
            self,
            cost_function,
            X, y,
            alpha=0.001,
            batch_size=16,
        ):
                    
        N = batch_size
        losses = []

        for idx in range(0, N, batch_size):

            batch_x = X[idx:idx+batch_size]
            batch_y = y[idx:idx+batch_size]
        
            y_pred = self.predict(batch_x)
            
            logit = batch_x.dot(self.W)
            w_hidden = self.hidden_layer_activation(logit)
            
            loss = cost_function(batch_y, y_pred)
            losses.append(loss)

            grad_erro = (batch_y - y_pred)
            
            erro_propagado = grad_erro.dot(self.V.T)
            
            Delta1 = erro_propagado * cost_function(w_hidden, mode='derivative')
            
            dEdV = grad_erro.T.dot(w_hidden)
            dEdW = batch_x.T.dot(Delta1) / N
            
            self.W -= alpha * dEdW
            self.V -= alpha * dEdV
                
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
            tol=0.001,
        ):
        """
        Função de treino para a MLP
        """
        print('='*10)
        print(f'Training classification model with {optimizer}')

        history_loss = []
        for epoch in range(epochs):
        
            if optimizer == 'GD':
                loss, dEdW, dEdV = self.gradient_descent(
                    cost_function=cost_function,
                    X=X_train, y=y_train,
                    alpha=learning_rate, n_epochs=epochs
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

            history_loss.append(loss)
            print(f'Avg training loss (GD)={np.mean(loss)}')
            print('Avg val cost function', val_loss)
        
        history_loss = np.array(history_loss)

        return history_loss

class RegLog():
    """ regressão logística"""
    def __init__(self):
        pass