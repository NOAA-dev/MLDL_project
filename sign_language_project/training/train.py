import numpy as np
from .loss import cross_entropy_loss
from .backward import backward
from .optimizer import SGD


def _batch_source(source):
    return source() if callable(source) else source


def _validate_batch(X, y, model):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 4:
        raise ValueError("Each X_batch must have NHWC shape (N,H,W,C).")
    if y.shape != (X.shape[0],):
        raise ValueError("Each y_batch must have shape (N,).")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y_batch must contain integer class IDs.")
    if X.shape[0] == 0:
        raise ValueError("Empty batches are not supported.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X_batch contains non-finite values.")
    if np.any(y < 0) or np.any(y >= model.cfg.num_classes):
        raise ValueError("y_batch contains a class ID outside the model's class range.")
    return X, y


def evaluate(model, batches):
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for X, y in _batch_source(batches):
        X, y = _validate_batch(X, y, model)
        probabilities = model.forward(X)
        batch_size = len(y)
        total_loss += cross_entropy_loss(probabilities, y) * batch_size
        total_correct += int(np.sum(np.argmax(probabilities, axis=1) == y))
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("No samples were provided for evaluation.")
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}


def train(model, train_batches, val_batches=None, epochs=30, learning_rate=0.01,
          optimizer=None, checkpoint_path=None, verbose=True, lr_schedule=None):
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    optimizer = optimizer or SGD(learning_rate)
    history = {"train_loss": [], "train_accuracy": []}
    if val_batches is not None:
        history.update({"val_loss": [], "val_accuracy": []})

    if lr_schedule is not None:
        history["learning_rate"] = []

    for epoch in range(epochs):
        if lr_schedule is not None:
            lr = float(lr_schedule(epoch))
            if not np.isfinite(lr) or lr <= 0:
                raise ValueError(f"lr_schedule returned an invalid learning rate {lr} at epoch {epoch}.")
            optimizer.learning_rate = lr
            history["learning_rate"].append(lr)
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for X, y in _batch_source(train_batches):
            X, y = _validate_batch(X, y, model)
            probabilities = model.forward(X)
            batch_size = len(y)
            batch_loss = cross_entropy_loss(probabilities, y)
            backward(model, probabilities, y)
            optimizer.step(model)
            total_loss += batch_loss * batch_size
            total_correct += int(np.sum(np.argmax(probabilities, axis=1) == y))
            total_samples += batch_size
        if total_samples == 0:
            raise ValueError("No training samples were provided for this epoch.")
        history["train_loss"].append(total_loss / total_samples)
        history["train_accuracy"].append(total_correct / total_samples)

        if val_batches is not None:
            metrics = evaluate(model, val_batches)
            history["val_loss"].append(metrics["loss"])
            history["val_accuracy"].append(metrics["accuracy"])

        if verbose:
            line = (f"Epoch {epoch + 1}/{epochs} - "
                    f"loss: {history['train_loss'][-1]:.4f} - "
                    f"accuracy: {history['train_accuracy'][-1]:.4f}")
            if val_batches is not None:
                line += (f" - val_loss: {history['val_loss'][-1]:.4f}"
                         f" - val_accuracy: {history['val_accuracy'][-1]:.4f}")
            if lr_schedule is not None:
                line += f" - lr: {history['learning_rate'][-1]:.5f}"
            print(line)

        if checkpoint_path is not None:
            from .checkpoints import save_checkpoint
            save_checkpoint(model, checkpoint_path, epoch=epoch + 1, history=history)
    return history
