import numpy as np
from utils.regularizations import lasso, ridge, elastic_net
from tqdm import tqdm

class MLP():
    """multi-layer perceptron"""
    def __init__(
            self,
            num_features,
            num_classes,
            num_neurons,
            hidden_layer_activation,
            output_layer_activation,
            cost_function,
            regularization = None,
        ):       
        self.M = num_features
        self.H = num_neurons # qtdade de neurônios na camada escondida
        self.K = num_classes # qtdade de classes para predição

        # W: matriz com pesos da camada de entrada
        self.W = np.random.randn(self.M, self.H) * 0.01 # shape: (M, H)
        # self.W = np.insert(W, 0, 0, axis=0) # add bias | shape: (M+1, H)
        # self.W = np.zeros((self.M + 1, self.H))
        # self.W[1:, :] = np.random.randn(self.M, self.H) * 0.01
        
        # V: matriz com pesos da camada de saida
        self.V = np.random.randn(self.H, self.K) * 0.01 # shape: (H, K)

        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation
        self.cost_function = cost_function

        self.regularization = regularization

        print(f'Instantiando MLP (regularização: {self.regularization})')


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
    
    def gradient_descent(
            self,
            X, y,
            alpha=0.001,
            batch_size=32,
        ):
                    
        N = X.shape[0]
        losses = []

        for idx in range(0, N, batch_size):

            batch_x = X[idx:idx+batch_size]
            batch_y = y[idx:idx+batch_size]
        
            """Forward pass"""
            logit = batch_x @ self.W                      # (N, H)
            hidden = self.hidden_layer_activation(logit) # (N, H)

            scores = hidden @ self.V                     # (N, K)
            y_pred = self.output_layer_activation(scores)
            
            loss = self.cost_function(batch_y, y_pred) # shape: (N, k)
            losses.append(loss)

            """Backprop"""
            grad_erro = (y_pred - batch_y) # shape: (N, k)
            
            erro_propagado = grad_erro.dot(self.V.T) # shape: (N, k) x (k, h) -> (N, h)
            
            Delta1 = \
                erro_propagado * self.hidden_layer_activation(
                    logit, derivative=True,
                ) # shape: (N, h) * (N, h) -> (N, h)
            
            dEdV = hidden.T @ grad_erro / batch_size # shape: (k, N) x (N, h) -> (k, h)
            dEdW = batch_x.T.dot(Delta1) / batch_size # shape: (m, N) x (N, h) -> (m, h)

            # Aplica regularização
            if self.regularization == 'l1':
                # TODO: consertar instabilidade (serrote)
                dEdW += lasso(self.W)
                dEdV += lasso(self.V, bias=False)
            
            elif self.regularization == 'l2':
                dEdW += ridge(self.W)
                dEdV += ridge(self.V, bias=False)

            elif self.regularization == 'elastic_net':
                dEdW += elastic_net(self.W)
                dEdV += elastic_net(self.V, bias=False)
            
            self.W -= alpha * dEdW # shape: (m, h) * (m, h)
            self.V -= alpha * dEdV # shape: (h, k) * (h, k)
                    
        losses = np.array(losses)

        return losses, dEdW, dEdV

    def fit(
            self,
            X_train, y_train, X_test, y_test,
            optimizer='GD',
            epochs=1000, 
            learning_rate=1e-3,
            tol=1e-2,
            kfold=3,
        ):
        """
        Função de treino para a MLP
        """
        print('='*10)
        print(f'Training classification model with {optimizer}')

        # Prepara dados de treino e testa com a adição do intercepto
        # X_train = np.insert(X_train, 0, 1, axis=1) # bias
        # X_test = np.insert(X_test, 0, 1, axis=1)

        fold_size = X_train.shape[0] // kfold

        kfold_loss_history = []
        for fold in range(kfold):
            print('Fold', fold+1)

            # re-inicializa pesos
            self.W = np.random.randn(*self.W.shape) * 0.01
            # self.W[1:, :] = np.random.randn(self.W.shape[0]-1, self.W.shape[1]) * 0.01
            # self.W[0, :] = 0.0
            self.V = np.random.randn(*self.V.shape) * 0.01

            start = fold *  fold_size
            end = (fold + 1) * fold_size \
                if fold != kfold - 1 else X_train.shape[0]

            mask = np.ones(len(X_train), dtype=bool)
            mask[list(range(start, end))] = False

            # seleciona partição de treino no KFold
            xtrain = X_train[mask]
            ytrain = y_train[mask]

            # seleciona partição de validação no KFold
            xval = X_train[start : end]
            yval = y_train[start : end]

            history_loss = []
            best_val_loss = float('inf')
            patience = 10  # parar se validação não melhorar por 10 épocas
            patience_counter = 0
            
            for epoch in tqdm(range(epochs)):
            
                if optimizer == 'GD':
                    loss, dEdW, dEdV = self.gradient_descent(
                        X=xtrain, y=ytrain,
                        alpha=learning_rate,
                        batch_size=32
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
                y_pred = self.predict(xval)

                val_loss = self.cost_function(yval, y_pred)

                # Early stopping: parar se validação não melhorar
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping na época {epoch}: validação não melhorou por {patience} épocas.')
                        break

                # Verifica a convergência baseado na norma dos gradientes
                if np.linalg.norm(dEdW) < tol and np.linalg.norm(dEdV) < tol: 
                    print(f'Early stopping na época {epoch}: gradientes menores que a tolerância')
                    break

                tqdm.write(f'epoch {epoch}: Avg training loss (GD)={np.mean(loss)} | val loss {val_loss}')
                
                history_loss.append([np.mean(loss), val_loss])
            
            kfold_loss_history.append(history_loss)
        
        y_pred = self.predict(X_test)
        test_loss = self.cost_function(y_test, y_pred)

        print('test loss:', test_loss)
        
        return kfold_loss_history