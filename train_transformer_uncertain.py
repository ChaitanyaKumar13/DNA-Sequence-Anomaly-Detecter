import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks
from utils.load_transformer_data import load_dataset
from transformer_dna_classifier import PositionalEncoding
from sklearn.model_selection import train_test_split
import os

# Configuration
MAX_LEN = 180
NUM_CLASSES = 3
EPOCHS = 30
BATCH_SIZE = 32
MODEL_SAVE_PATH = "models/transformer_dna_uncertain.keras"
DATA_PATH = "data/dna.csv"
TRAIN_PATH = "data/dna_train.csv"
VAL_PATH = "data/dna_valid.csv"

# Enable dropout in transformer blocks
def build_model():
    inputs = layers.Input(shape=(MAX_LEN, 1), name="input_layer")
    x = layers.Dense(64)(inputs)
    x = PositionalEncoding(max_len=MAX_LEN, embed_dim=64)(x)

    for _ in range(2):
        x1 = layers.LayerNormalization()(x)
        attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.3)(x1, x1)
        x2 = layers.Add()([attn_out, x])
        x3 = layers.LayerNormalization()(x2)
        ff = layers.Dense(128, activation='relu')(x3)
        ff = layers.Dense(64)(ff)
        x = layers.Add()([x2, ff])
        x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

# Step 1: Read and split the full dataset
df = pd.read_csv(DATA_PATH).dropna()
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])

# Step 2: Save train and valid files
df_train.to_csv(TRAIN_PATH, index=False)
df_val.to_csv(VAL_PATH, index=False)

# Step 3: Delete original dna.csv to avoid confusion
if os.path.exists(DATA_PATH):
    os.remove(DATA_PATH)
    print(f"✅ Deleted original dataset: {DATA_PATH}")

# Step 4: Load processed data
X_train, y_train = load_dataset(TRAIN_PATH, max_len=MAX_LEN, one_hot=True, add_noise=True)
X_val, y_val = load_dataset(VAL_PATH, max_len=MAX_LEN, one_hot=True)

# Step 5: Build and train model
model = build_model()
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]
)
