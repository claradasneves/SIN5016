import numpy as np
from utils.regularizations import lasso, ridge, elastic_net

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

        self.W = np.random.randn(self.M, self.K)

        self.cost_function = cost_function
        self.regularization = regularization

        print(f'Instantiando Regressão Logística Multinomial (regularização: {self.regularization})')

    def predict(self, X: np.array):
        """Modelo da Regressão Logística multinomial"""
        
        logit = X.dot(self.W)

        logit_max = np.max(logit, axis=1, keepdims=True)
        logit_estavel = logit - logit_max
        
        exp_logit = np.exp(logit_estavel)

        exp_categorias = np.sum(exp_logit, axis=1, keepdims=True)
        
        return exp_logit / exp_categorias

    def gradient_descent(
            self,
            X, y,
            alpha=0.001,
            batch_size=64,
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

            if self.regularization == 'l1':
                dEdW += lasso(self.W, bias=False)
            
            elif self.regularization == 'l2':
                dEdW += ridge(self.W, bias=False)
            
            elif self.regularization == 'elastic_net':
                dEdW += elastic_net(self.W, bias=False)
            
            self.W -= alpha * dEdW
        
        losses = np.array(losses)

        return losses, dEdW
    

    def fit(
            self,
            X_train, y_train, X_test, y_test,
            optimizer='GD',
            epochs=1000, 
            learning_rate=0.0001,
            tol=0.01,
            kfold=3,
        ):
        """
        Função de treino para a MLP
        """
        print('='*10)
        print(f'Training classification model with {optimizer}')

        fold_size = X_train.shape[0] // kfold

        kfold_loss_history = []
        for fold in range(kfold):
            print('Fold', fold+1)

            # re-inicializa pesos
            self.W = np.random.randn(*self.W.shape) * 0.001

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
            patience = 100  # parar se validação não melhorar por 10 épocas
            patience_counter = 0
            
            for epoch in range(epochs):
            
                if optimizer == 'gd':
                    loss, dEdW = self.gradient_descent(
                        X=xtrain, y=ytrain,
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
                if np.linalg.norm(dEdW) < tol: 
                    print(f'Convergência atingida no Gradiente | Época {epoch}')
                    break

                # print(f'epoch {epoch}: Avg training loss (GD)={np.mean(loss)} | val loss {val_loss}')
                
                history_loss.append([np.mean(loss), val_loss])
            
            kfold_loss_history.append(history_loss)
        
        # avalia erro no conjunto de teste - nunca visto
        y_pred = self.predict(X_test)
        test_loss = self.cost_function(y_test, y_pred)

        print('test loss:', test_loss)
        
        return kfold_loss_history
    