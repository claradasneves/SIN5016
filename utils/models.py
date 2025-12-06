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
        self.W = np.random.randn(self.M, self.H) * 0.001 # shape: (M, H)
        # TODO: self.W = np.insert(W, 0, 1, axis=1) # add bias
        
        # V: matriz com pesos da camada de saida
        self.V = np.random.randn(self.H, self.K) * 0.001 # shape: (H, K)
        # TODO: self.V = np.insert(V, 0, 1, axis=1) # add bias

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

            # TODO: adiciona intercepto
            # batch_x = np.insert(batch_x, 0, 1, axis=1)
        
            y_pred = self.predict(batch_x) # shape: (N, k)
            
            logit = batch_x.dot(self.W) # shape: (N, m) x (m, h) -> (N, h)
            w_hidden = self.hidden_layer_activation(logit) # shape: (N, h)
            
            loss = cost_function(batch_y, y_pred) # shape: (N, k)
            losses.append(loss)

            grad_erro = (batch_y - y_pred) # shape: (N, k)
            
            erro_propagado = grad_erro.dot(self.V.T) # shape: (N, k) x (k, h) -> (N, h)
            
            Delta1 = \
                erro_propagado * self.hidden_layer_activation(
                    w_hidden, derivative=True,
                ) # shape: (N, h) * (N, h) -> (N, h)
            
            dEdV = grad_erro.T.dot(w_hidden) # shape: (k, N) x (N, h) -> (k, h)
            dEdW = batch_x.T.dot(Delta1) / N # shape: (m, N) x (N, h) -> (m, h)
            
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

            history_loss.append(loss)
            print(f'epoch {epoch}: Avg training loss (GD)={np.mean(loss)} | val loss {val_loss}')
        
        history_loss = np.array(history_loss)

        return history_loss

class RegLog():
    """ regressão logística"""
    def __init__(
            self,
            num_features,
        ):
        self.M = num_features

        W = np.random.randn(self.M, 1)
        self.W = np.insert(W, 0, 1, axis=1) # add bias

    def gradient_descent(
            self,
            cost_function,
            X, y,
            alpha=0.001, 
            tol=0.01,
            batch_size=16):
        """"
        Args:
            x: input vector
            steps: qtdade de passos para andar na direção do gradiente
            alpha: escalar que reduz o tamanho do vetor gradiente, isso serve para eu saber a direção, mas andar passos curtos na direção indicada
            tol: o tamanho mínimo que eu espero que meu gradiente chegue. Isso indica o tamanho de tolerância para atualizações do gradiente.
        
        Returns
            o ponto mínimo local, no contexto dos hiperparâmetros.
        """

        N = batch_size
        losses = []

        for start in range(0, N, batch_size):

            batch_x = X[start:start+batch_size]
            batch_y = y[start:start+batch_size]
            
            y_pred = self.predict(batch_x)

            loss = cost_function(batch_y, y_pred)
            losses.append(loss)

            # shape: (N, m).T -> (m, N) x (N, k) -> (m, k)
            dEdW = batch_x.T.dot(y_pred - batch_y)
            
            self.W -= alpha * dEdW
        
        losses = np.array(losses)

        return losses, dEdW
    
    def predict(self, X: np.array):
        """Modelo da Regressão Logística"""
        
        logits = X.dot(self.W)
        logistica = (np.e**logits) / (1 + np.e**logits)
        
        return logistica
    
    def fit(
        self,
        cost_function,
        X_train, y_train, X_test, y_test,
        optimizer='GD',
        epochs=1000, 
        learning_rate=0.001,
        tol=0.001,    
        ):

        history_loss = []
        for epoch in range(epochs):

            if optimizer == 'GD':
                losses, dEdW = self.gradient_descent(
                    X=X_train, 
                    y=y_train,
                    cost_function=cost_function,
                    alpha=learning_rate,
                )

            # Avaliação Final
            y_pred_val = self.predict(X_test)

            val_loss = cost_function(y_test, y_pred_val)
            history_loss.append(val_loss)

            if np.linalg.norm(dEdW) < tol: break

            print(f'Avg training loss={np.mean(losses)}')
            print(f'Avg validation loss={np.mean(history_loss)}')

        history_loss = np.array(history_loss)

        return history_loss
    