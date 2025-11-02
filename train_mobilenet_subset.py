"""
Train a MobileNetV2-based deepfake detector on the subset faces using transfer learning.

Highlights:
- Input size 160x160 for speed on CPU/GPU
- Freeze base then optionally fine-tune top layers
- AUC-based early stopping and checkpointing
- Class weights for imbalance

Artifacts:
- deepfake_detector_mobilenet_subset.keras
- deepfake_detector_mobilenet_subset.weights.h5
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS_FREEZE = 15
EPOCHS_FINETUNE = 5  # keep small to avoid overfitting on small subset


def build_model():
    base = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False  # freeze initially

    x = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    y = base(x, training=False)
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dropout(0.3)(y)
    out = layers.Dense(1, activation='sigmoid')(y)
    model = keras.Model(inputs=x, outputs=out)
    return model, base


def compute_class_weight(train_df: pd.DataFrame):
    counts = train_df['label'].value_counts()
    real = int(counts.get('REAL', 0))
    fake = int(counts.get('FAKE', 0))
    total = max(1, real + fake)
    return {
        0: float(total / (2.0 * max(1, real))),  # REAL -> 0
        1: float(total / (2.0 * max(1, fake))),  # FAKE -> 1
    }


def main():
    print("=" * 70)
    print("MOBILENETV2 DEEPFAKE DETECTOR - transfer learning on subset")
    print("=" * 70)

    splits_dir = Path('splits_subset')
    train_csv = splits_dir / 'train.csv'
    val_csv = splits_dir / 'val.csv'
    if not (train_csv.exists() and val_csv.exists()):
        print('ERROR: splits_subset not found. Build subset first.')
        return

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    print('Train dist:\n', train_df['label'].value_counts())
    print('Val dist:\n', val_df['label'].value_counts())

    class_weight = compute_class_weight(train_df)
    print('Class weights (0=REAL,1=FAKE):', class_weight)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=(0.8, 1.2),
        shear_range=0.1,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', classes=['REAL', 'FAKE'], shuffle=True
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col='filepath', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', classes=['REAL', 'FAKE'], shuffle=False
    )
    print('Class mapping:', train_gen.class_indices)

    model, base = build_model()
    model.compile(
        optimizer=Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
    )

    os.makedirs('checkpoints', exist_ok=True)
    callbacks = [
        ModelCheckpoint('checkpoints/mobilenet_freeze_best.weights.h5', monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]

    print("\n== Stage 1: Train with frozen base ==")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FREEZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Optional fine-tuning of top layers if validation AUC is decent
    print("\n== Stage 2: Fine-tune top layers ==")
    # Unfreeze last N layers (excluding BatchNorm layers which are best kept frozen)
    unfreeze_from = max(0, len(base.layers) - 50)
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= unfreeze_from) and not isinstance(layer, layers.BatchNormalization)

    model.compile(
        optimizer=Adam(1e-4),  # lower LR for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
    )

    callbacks_ft = [
        ModelCheckpoint('checkpoints/mobilenet_ft_best.weights.h5', monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=1, min_lr=1e-5, verbose=1)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weight,
        callbacks=callbacks_ft,
        verbose=1
    )

    print("\n== Final evaluation ==")
    results = model.evaluate(val_gen, verbose=0)
    metrics = dict(zip(model.metrics_names, results))
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save artifacts
    weights_out = 'deepfake_detector_mobilenet_subset.weights.h5'
    keras_out = 'deepfake_detector_mobilenet_subset.keras'
    model.save_weights(weights_out)
    print(f"Saved weights -> {weights_out}")
    try:
        model.save(keras_out, include_optimizer=False)
        print(f"Saved model   -> {keras_out}")
    except Exception as e:
        print(f"Warning: could not save .keras: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
