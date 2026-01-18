import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tqdm import tqdm


class CNNModel:
    """Rede Neural Convolucional."""
    
    def __init__(self, input_shape=(64, 64, 3), num_classes=50):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def _build_model(self):
        """Construcao da arquitetura do modelo"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
                
        # First block
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Second block
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Third block
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def fit(self, X_train, y_train, X_test, y_test,
            learning_rate=1e-3, epochs=100, batch_size=32, 
            validation_split=0.1):
        """
        Treina a CNN com os dados fornecidos.

        Args:
            X_train: imagens de treino (N, H, W, C)
            y_train: labels de treino - one-hot (N, num_classes) ou indices (N,)
            X_test: imagens de teste
            y_test: labels de teste
            learning_rate: taxa de aprendizado
            epochs: número de épocas
            batch_size: tamanho do batch
            validation_split: parte dos dados de treino para validação

        Retorna:
            tupla (history_list, test_loss).
        """
        # constroi modelo
        self.model = self._build_model()

        save_best_model = callbacks.ModelCheckpoint(
            filepath=f'experiments/weights/cnn_checkpoint.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0,
        )
        callback = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
        )
        
        # Detectar formato: one-hot (2D) ou indices (1D)
        if y_train.ndim == 2:
            # one-hot encoded
            loss_fn = 'categorical_crossentropy'
            y_train_to_fit = y_train
            y_test_to_fit = y_test
        else:
            # numeric indices
            loss_fn = 'sparse_categorical_crossentropy'
            y_train_to_fit = y_train
            y_test_to_fit = y_test
        
        # compila modelo
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=['accuracy']
        )

        # visualiza arquitetura da cnn
        self.model.summary()
        
        # treina modelo
        self.history = self.model.fit(
            X_train, 
            y_train_to_fit,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[callback, save_best_model],
            verbose=1,
        )
        
        # evaluate no conjunto de teste
        test_loss, test_acc = self.model.evaluate(X_test, y_test_to_fit, verbose=0)
        print(f'\nTest accuracy: {test_acc:.4f}')
        print(f'Test loss: {test_loss:.4f}')
                
        return self.history, test_loss

    
    def predict(self, X):
        """
        Faz o predict com o modelo treinado.
        
        Args:
            X: imagens de input (N, H, W, C)
            
        Retorna:
            predictions: probabilidade das classes (N, num_classes)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo nos dados de teste.
        
        Args:
            X_test: imagens de teste
            y_test: labels de teste (indices numéricos)
            
        Retorna:
            loss: erro de teste
            accuracy: acurácia de teste
        """
        y_test_indices = y_test
        
        loss, accuracy = self.model.evaluate(X_test, y_test_indices, verbose=0)
        return loss, accuracy
    
    def save(self, filepath):
        """Salva o modelo em um arquivo."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.model.save(filepath)
    
    def load(self, filepath):
        """Carrega o modelo de um arquivo."""
        self.model = models.load_model(filepath)

