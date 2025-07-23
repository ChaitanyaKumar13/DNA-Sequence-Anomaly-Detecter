import pandas as pd
import numpy as np
import tensorflow as tf

def load_dataset(filepath, max_len=180, one_hot=False, add_noise=False):
    """
    Loads and preprocesses DNA sequence data for Transformer model.

    Args:
        filepath (str): Path to the CSV file.
        max_len (int): Expected sequence length (number of binary features).
        one_hot (bool): If True, returns one-hot encoded class labels.
        add_noise (bool): If True, adds Gaussian noise to input features.

    Returns:
        Tuple: (X, y)
            X: np.ndarray of shape (num_samples, max_len, 1)
            y: np.ndarray of shape (num_samples,) or (num_samples, num_classes)
    """
    # Load data and drop missing rows
    df = pd.read_csv(filepath).dropna()

    # Extract feature columns that start with "A"
    feature_cols = [col for col in df.columns if col.startswith("A")]
    if len(feature_cols) != max_len:
        raise ValueError(f"Expected {max_len} feature columns but got {len(feature_cols)}. Check your CSV format.")

    # Extract features
    X_raw = df[feature_cols].astype(np.float32).values

    # Optionally add Gaussian noise (for uncertainty training)
    if add_noise:
        noise = np.random.normal(loc=0.0, scale=0.05, size=X_raw.shape)
        X_raw = np.clip(X_raw + noise, 0.0, 1.0)  # Ensure values stay in [0, 1]

    # Reshape to 3D input for Conv/Transformer layers
    X_raw = X_raw.reshape((-1, max_len, 1))

    # Map labels to integers: 1=EI (0), 2=IE (1), 3=Neither (2)
    y = df['class'].apply(lambda v: {1: 0, 2: 1, 3: 2}[v]).values

    # One-hot encode if specified
    if one_hot:
        y = tf.keras.utils.to_categorical(y, num_classes=3)

    return X_raw, y
