from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

class ModelTrainer:
    def __init__(self, model, model_name="model.h5"):
        self.model = model
        self.model_name = model_name

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, save_dir="saved_models"):
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, self.model_name)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history
