import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
import pickle
from threading import Thread, Event
import time
from scipy import signal

# ================== GPU Setup ==================
class GPUManager:
    @staticmethod
    def setup_gpu():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                gpu = gpus[0]
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]
                )
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"GPU: {gpu.name} | Memory: 10GB | FP16: Enabled")
                return True
            except Exception as e:
                print(f"GPU setup failed: {e}")
                return False
        return False

# ================== Vacuum Range Manager ==================
class VacuumRangeManager:
    RANGES = {
        'ULTRA_HIGH': {'min': 1.0e-5, 'max': 4.9e-5, 'color': '#2E7D32'},
        'VERY_HIGH': {'min': 5.0e-5, 'max': 9.9e-5, 'color': '#1976D2'},
        'HIGH': {'min': 1.0e-4, 'max': 4.9e-4, 'color': '#F57C00'},
        'MEDIUM': {'min': 5.0e-4, 'max': 9.9e-4, 'color': '#C62828'}
    }
    
    @classmethod
    def get_range_name(cls, vacuum_value):
        for range_id, info in cls.RANGES.items():
            if info['min'] <= vacuum_value <= info['max']:
                return range_id
        return 'ULTRA_HIGH' if vacuum_value < cls.RANGES['ULTRA_HIGH']['min'] else 'MEDIUM'

# ================== Custom Callbacks ==================
class StopTrainingCallback(keras.callbacks.Callback):
    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event
    
    def on_batch_end(self, batch, logs=None):
        if self.stop_event.is_set():
            self.model.stop_training = True

class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, gui_callback, total_epochs):
        super().__init__()
        self.gui_callback = gui_callback
        self.total_epochs = total_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        self.gui_callback(epoch + 1, logs, ((epoch) / self.total_epochs) * 100, 'begin')
    
    def on_epoch_end(self, epoch, logs=None):
        progress = ((epoch + 1) / self.total_epochs) * 100
        self.gui_callback(epoch + 1, logs, progress, 'end')

# ================== LSTM_IF ensemble Model ==================
class EnhancedLSTMIFModel:
    def __init__(self, sequence_length=700, n_features=7,
                 encoder_units=[64, 32], latent_dim=16,
                 decoder_units=[32, 64],
                 dropout=0.25, recurrent_dropout=0.15,
                 lr=0.001, use_bidirectional=True,
                 l2_reg=0.001, clip_norm=1.0,
                 optimizer='adam',
                 latent_activation='tanh',
                 use_batch_norm=True):
        
        # ✅ 반드시 build_model() 호출 전에 모든 속성 저장
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        self.encoder_units = encoder_units
        self.latent_dim = latent_dim
        self.decoder_units = decoder_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.lr = lr
        self.use_bidirectional = use_bidirectional
        self.l2_reg = l2_reg
        self.clip_norm = clip_norm
        self.optimizer = optimizer
        self.latent_activation = latent_activation
        self.use_batch_norm = use_batch_norm
        
        # ✅ 이제 build_model() 호출 가능
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        
        self.threshold = None
        self.vacuum_range_thresholds = {}
        self.scalers = {}
        self.history = {}
        self.training_time = 0
        self.best_epoch = 0
        
        self.train_stats = {}
        self.val_stats = {}
        
        self.model_id = None
        self.created_at = None
        
        self.build_model()
    
    @staticmethod
    @tf.function
    def weighted_mse(y_true, y_pred):
        weights = tf.constant([2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.5], dtype=tf.float32)
        squared_diff = tf.square(y_true - y_pred)
        weighted_diff = squared_diff * tf.reshape(weights, (1, 1, 7))
        return tf.reduce_mean(weighted_diff)
    
    def build_encoder(self):
        inputs = keras.Input(shape=(self.sequence_length, self.n_features), dtype='float32')
        x = inputs
        
        for i, units in enumerate(self.encoder_units):
            return_sequences = i < len(self.encoder_units) - 1
            
            if self.use_bidirectional:
                x = layers.Bidirectional(
                    layers.LSTM(units, return_sequences=return_sequences)
                )(x)
            else:
                x = layers.LSTM(units, return_sequences=return_sequences)(x)
            
            if self.use_batch_norm:  # ✅ 조건부 적용
                x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)
        
        latent = layers.Dense(
            self.latent_dim,
            activation=self.latent_activation,
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='latent'
        )(x)
        
        return Model(inputs, latent, name='Encoder')
    
    def build_decoder(self):
        latent_input = keras.Input(shape=(self.latent_dim,), dtype='float32')
        
        x = layers.Dense(self.decoder_units[0])(latent_input)
        if self.use_batch_norm:  # ✅ 조건부 적용
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout)(x)
        
        x = layers.RepeatVector(self.sequence_length)(x)
        
        for i, units in enumerate(self.decoder_units):
            if self.use_bidirectional:
                x = layers.Bidirectional(
                    layers.LSTM(units, return_sequences=True)
                )(x)
            else:
                x = layers.LSTM(units, return_sequences=True)(x)
            
            if self.use_batch_norm:  # ✅ 조건부 적용
                x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)
        
        outputs = layers.TimeDistributed(
            layers.Dense(self.n_features, dtype='float32')
        )(x)
        
        return Model(latent_input, outputs, name='Decoder')
    
    def build_model(self):
        with tf.device('/GPU:0'):
            self.encoder = self.build_encoder()
            self.decoder = self.build_decoder()
            
            ae_input = keras.Input(shape=(self.sequence_length, self.n_features), dtype='float32')
            latent = self.encoder(ae_input)
            ae_output = self.decoder(latent)
            
            self.autoencoder = Model(ae_input, ae_output, name='EnhancedLSTMAE')
            
            # ✅ Optimizer 선택 로직 추가
            if self.optimizer == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=self.clip_norm)
            elif self.optimizer == 'sgd':
                optimizer = keras.optimizers.SGD(learning_rate=self.lr, momentum=0.9, 
                                                nesterov=True, clipnorm=self.clip_norm)
            elif self.optimizer == 'rmsprop':
                optimizer = keras.optimizers.RMSprop(learning_rate=self.lr, rho=0.9, 
                                                     clipnorm=self.clip_norm)
            elif self.optimizer == 'adamax':
                optimizer = keras.optimizers.Adamax(learning_rate=self.lr, clipnorm=self.clip_norm)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")
            
            if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            
            self.autoencoder.compile(
                optimizer=optimizer,
                loss=EnhancedLSTMIFModel.weighted_mse,
                metrics=['mae', 'mse']
            )
    
    def preprocess_data(self, data):
            df = pd.DataFrame(data)
            
            # 🔧 정상 + 이상 탐지 범위 포함 Clipping
            # 정상: Vacuum 1e-5~9e-3, Temp 10~15K, Press 1.8~2.1MPa
            # 이상: Vacuum >9e-3 (최대 ~2e-2), Temp >15K 상승, Press 이탈
            bounds = [
                (5e-6, 0.025),     # Vacuum (Pa) - 정상 전체 + 이상 범위 포함
                (8.0, 25.0),       # Pump1 Temp (K) - 정상 + 이상 상승 범위
                (8.0, 25.0),       # Pump2 Temp (K)
                (8.0, 25.0),       # Pump3 Temp (K)
                (8.0, 25.0),       # Pump4 Temp (K)
                (1.2, 2.8),        # Comp1 Press (MPa) - 정상 + 이탈 여유
                (1.2, 2.8),        # Comp2 Press (MPa)
            ]
            
            for col in range(self.n_features):
                low, high = bounds[col]
                df.iloc[:, col] = np.clip(df.iloc[:, col], low, high)
            
            data = df.values.astype(np.float32)
            
            # 전처리된 데이터 저장
            processed = np.zeros_like(data, dtype=np.float32)
            
            # 1. 진공도 처리 (Log + RobustScaler)
            vacuum_log = np.log10(np.clip(data[:, 0], 1e-10, 1))
            if 'vacuum' not in self.scalers:
                self.scalers['vacuum'] = RobustScaler(quantile_range=(5, 95))  # 더 보수적으로
                processed[:, 0] = self.scalers['vacuum'].fit_transform(vacuum_log.reshape(-1, 1)).flatten()
            else:
                processed[:, 0] = self.scalers['vacuum'].transform(vacuum_log.reshape(-1, 1)).flatten()
            
            # 2. 온도/압력 처리 (RobustScaler로 변경 - 이상치에 강함)
            for i in range(1, self.n_features):
                if f'sensor_{i}' not in self.scalers:
                    # 🔧 StandardScaler → RobustScaler 변경
                    self.scalers[f'sensor_{i}'] = RobustScaler(quantile_range=(5, 95))
                    processed[:, i:i+1] = self.scalers[f'sensor_{i}'].fit_transform(data[:, i:i+1])
                else:
                    processed[:, i:i+1] = self.scalers[f'sensor_{i}'].transform(data[:, i:i+1])
            
            # 3. Savitzky-Golay 필터 적용 (진공도 제외, 온도/압력만)
            for i in range(1, self.n_features):
                window = min(51, len(processed)//10*2+1)
                if window % 2 == 0:
                    window += 1
                if window >= 5:
                    processed[:, i] = signal.savgol_filter(processed[:, i], window, min(3, window-2))
            
            return processed
    
    def downsample(self, data):
        return data.reshape(-1, 10, self.n_features).mean(axis=1)
    
    def augment_data(self, data, n_augmented=3):
        augmented = [data]
        
        for i in range(n_augmented):
            noise_level = 0.002 * (i + 1)
            noise = np.random.normal(0, noise_level, data.shape)
            augmented.append(data + noise)
            
            if i < 2:
                scale = np.random.uniform(0.98, 1.02)
                augmented.append(data * scale)
        
        return augmented
    
    def train(self, train_data, val_data, epochs=100, batch_size=32,
             augmentation=3, patience=20, min_delta=0.0001,
             lr_schedule='plateau', save_checkpoints=True,
             checkpoint_dir=None, stop_event=None,
             progress_callback=None, verbose=1):
        
        start_time = time.time()
        
        # 데이터 전처리
        X_train = []
        for data in train_data:
            processed = self.preprocess_data(data)
            downsampled = self.downsample(processed)
            augmented = self.augment_data(downsampled, augmentation)
            X_train.extend(augmented)
        X_train = np.array(X_train, dtype=np.float32)
        
        X_val = []
        for data in val_data:
            processed = self.preprocess_data(data)
            downsampled = self.downsample(processed)
            X_val.append(downsampled)
        X_val = np.array(X_val, dtype=np.float32)
        
        # Calculate actual validation split
        total_samples = len(train_data) + len(val_data)
        actual_val_split = len(val_data) / total_samples if total_samples > 0 else 0.0
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, X_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Learning Rate Scheduler
        if lr_schedule == 'plateau':
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            ))
        elif lr_schedule == 'exponential':
            def exp_decay(epoch):
                return self.lr * 0.95 ** epoch
            callbacks.append(keras.callbacks.LearningRateScheduler(exp_decay, verbose=1))
        elif lr_schedule == 'cosine':
            callbacks.append(keras.callbacks.LearningRateScheduler(
                lambda epoch: self.lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs)),
                verbose=1
            ))
        
        # Checkpoint Callback
        if save_checkpoints and checkpoint_dir:
            checkpoint_path = checkpoint_dir / f"checkpoint_{{epoch:03d}}_{{val_loss:.6f}}.h5"
            callbacks.append(keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ))
        
        # Stop Training Callback
        if stop_event:
            callbacks.append(StopTrainingCallback(stop_event))
        
        # Progress Callback
        if progress_callback:
            callbacks.append(ProgressCallback(progress_callback, epochs))
        
        print(f"\n{'='*60}")
        print(f"Training Enhanced LSTM-AE Model")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        print(f"Learning Rate Schedule: {lr_schedule}")
        print(f"{'='*60}")
        
        history = self.autoencoder.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        self.training_time = time.time() - start_time
        self.best_epoch = np.argmin(history.history['val_loss']) + 1
        
        # Store complete training parameters - 모든 파라미터를 저장
        self.training_params = {
            'epochs': epochs,
            'epochs_trained': len(self.history.get('loss', [])),
            'batch_size': batch_size,
            'augmentation': augmentation,
            'patience': patience,
            'min_delta': min_delta,
            'lr_schedule': lr_schedule,
            'validation_split': actual_val_split,
            'save_checkpoints': save_checkpoints,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'augmented_train_samples': len(X_train),
            'processed_val_samples': len(X_val),
            'optimizer': self.optimizer
        }
        
        self.calculate_statistics(X_train, X_val)
        self.calculate_threshold(X_train, X_val, train_data)
        
        self.created_at = datetime.now()
        self.model_id = f"unified_{self.created_at.strftime('%y%m%d_%H%M')}"
        
        return self.history
    
    def calculate_statistics(self, train_data, val_data):
        train_pred = self.autoencoder.predict(train_data, verbose=0)
        val_pred = self.autoencoder.predict(val_data, verbose=0)
        
        train_mse = np.mean(np.square(train_data - train_pred), axis=(1, 2))
        val_mse = np.mean(np.square(val_data - val_pred), axis=(1, 2))
        
        self.train_stats = {
            'mean': float(np.mean(train_mse)),
            'std': float(np.std(train_mse)),
            'min': float(np.min(train_mse)),
            'max': float(np.max(train_mse)),
            'q25': float(np.percentile(train_mse, 25)),
            'q50': float(np.percentile(train_mse, 50)),
            'q75': float(np.percentile(train_mse, 75)),
            'q95': float(np.percentile(train_mse, 95)),
            'q99': float(np.percentile(train_mse, 99))
        }
        
        self.val_stats = {
            'mean': float(np.mean(val_mse)),
            'std': float(np.std(val_mse)),
            'min': float(np.min(val_mse)),
            'max': float(np.max(val_mse)),
            'q25': float(np.percentile(val_mse, 25)),
            'q50': float(np.percentile(val_mse, 50)),
            'q75': float(np.percentile(val_mse, 75)),
            'q95': float(np.percentile(val_mse, 95)),
            'q99': float(np.percentile(val_mse, 99))
        }
    
    def calculate_threshold(self, train_data, val_data, original_train_data):
        train_pred = self.autoencoder.predict(train_data, verbose=0)
        val_pred = self.autoencoder.predict(val_data, verbose=0)
        
        weights = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.5])
        train_errors = np.mean(np.square(train_data - train_pred) * weights, axis=(1, 2))
        val_errors = np.mean(np.square(val_data - val_pred) * weights, axis=(1, 2))
        
        all_errors = np.concatenate([train_errors, val_errors])
        
        median = np.median(all_errors)
        mad = np.median(np.abs(all_errors - median))
        
        base_multiplier = 5.5
        total = len(train_data) + len(val_data)
        if total < 500:
            base_multiplier += 0.4
        elif total > 1500:
            base_multiplier -= 0.1
        
        self.threshold = median + base_multiplier * mad
        min_threshold = np.percentile(all_errors, 98)
        self.threshold = max(self.threshold, min_threshold)
        
        self._calculate_range_adjustments(original_train_data, train_errors)
        
        return self.threshold
    
    def _calculate_range_adjustments(self, original_data, errors):
        range_errors = {k: [] for k in VacuumRangeManager.RANGES.keys()}
        
        for data, error in zip(original_data, errors):
            vacuum = float(data[0, 0])
            range_name = VacuumRangeManager.get_range_name(vacuum)
            range_errors[range_name].append(error)
        
        for range_name, errs in range_errors.items():
            if len(errs) > 10:
                range_median = np.median(errs)
                adjustment = range_median / np.median(errors)
                self.vacuum_range_thresholds[range_name] = float(adjustment)
            else:
                self.vacuum_range_thresholds[range_name] = 1.0
    
    def detect_anomaly(self, data, return_details=True):
        processed = self.preprocess_data(data)
        downsampled = self.downsample(processed)
        downsampled = downsampled.reshape(1, self.sequence_length, self.n_features)
        
        prediction = self.autoencoder.predict(downsampled, verbose=0)
        
        errors = np.square(downsampled - prediction)
        mse = np.mean(errors)
        
        weights = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.5])
        weighted_error = np.mean(errors * weights)
        
        sensor_errors = np.mean(errors, axis=(0, 1))
        if sensor_errors.ndim > 1:
            sensor_errors = sensor_errors.flatten()
        
        vacuum = float(data[0, 0])
        vacuum_range = VacuumRangeManager.get_range_name(vacuum)
        
        range_adjustment = self.vacuum_range_thresholds.get(vacuum_range, 1.0)
        adjusted_threshold = self.threshold * range_adjustment
        
        anomaly_score = weighted_error
        
        is_anomaly = anomaly_score > adjusted_threshold
        confidence = min(1.0, abs(anomaly_score - adjusted_threshold) / adjusted_threshold)
        
        time_errors = np.mean(errors[0], axis=1)
        peak_time = int(np.argmax(time_errors))
        
        sensor_names = ['Vacuum', 'Pump1_Temp', 'Pump2_Temp', 'Pump3_Temp',
                       'Pump4_Temp', 'Comp1_Press', 'Comp2_Press']
        contributions = sensor_errors / (np.sum(sensor_errors) + 1e-8)
        anomaly_sensors = [sensor_names[i] for i, c in enumerate(contributions) if c > 0.2]
        
        result = {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'threshold': float(self.threshold),
            'adjusted_threshold': float(adjusted_threshold),
            'confidence': float(confidence),
            'vacuum_range': vacuum_range,
            'vacuum_value': float(vacuum),
            'mse': float(mse),
            'weighted_error': float(weighted_error),
            'peak_time': peak_time
        }
        
        if return_details:
            result.update({
                'sensor_errors': sensor_errors.tolist(),
                'sensor_contributions': contributions.tolist(),
                'anomaly_sensors': anomaly_sensors,
                'time_errors': time_errors.tolist()
            })
        
        return result
    
    def save(self, base_path, ensemble_enabled=False):
        """Save model with timestamp in filename"""
        save_dir = Path(base_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.created_at.strftime('%y%m%d_%H%M')
        if ensemble_enabled:
            save_name = f"lstmif_ensemble_{timestamp}.h5"
        else:
            save_name = f"lstmae_only_{timestamp}.h5"
        full_path = save_dir / save_name
        
        self.autoencoder.save(str(full_path))
        
        # 완전한 메타데이터 저장
        metadata = {
            'model_type': 'unified',
            'model_id': self.model_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'threshold': self.threshold,
            'vacuum_range_thresholds': self.vacuum_range_thresholds,
            'params': {
                'encoder_units': self.encoder_units,
                'latent_dim': self.latent_dim,
                'decoder_units': self.decoder_units,
                'dropout': self.dropout,
                'recurrent_dropout': self.recurrent_dropout,
                'lr': self.lr,
                'use_bidirectional': self.use_bidirectional,
                'optimizer': self.optimizer,
                'latent_activation': self.latent_activation,
                'use_batch_norm': self.use_batch_norm 
            },
            'training_params': getattr(self, 'training_params', {}),
            'history': self.history,
            'train_stats': self.train_stats,
            'val_stats': self.val_stats,
            'training_time': self.training_time,
            'best_epoch': self.best_epoch
        }
        
        prefix = "lstmif_ensemble" if ensemble_enabled else "lstmae_only"
        with open(save_dir / f"{prefix}_{timestamp}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        with open(save_dir / f"{prefix}_{timestamp}_scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        return full_path
    
    def load(self, path):
        model_path = Path(path)
        
        # ✅ 1단계: metadata 먼저 로드
        stem = model_path.stem
        parts = stem.split('_')
        if len(parts) >= 3:
            timestamp_str = f"{parts[-2]}_{parts[-1]}"
        
        metadata_path = model_path.parent / f"{stem}_metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # ✅ 2단계: 파라미터를 먼저 self에 저장 (모델 로드 전)
        params = metadata.get('params', {})
        self.encoder_units = params.get('encoder_units', [64, 32])
        self.latent_dim = params.get('latent_dim', 16)
        self.decoder_units = params.get('decoder_units', [32, 64])
        self.dropout = params.get('dropout', 0.25)
        self.recurrent_dropout = params.get('recurrent_dropout', 0.15)
        self.lr = params.get('lr', 0.001)
        self.use_bidirectional = params.get('use_bidirectional', True)
        self.optimizer = params.get('optimizer', 'adam')
        self.latent_activation = params.get('latent_activation', 'tanh')  # ✅ 여기!
        self.use_batch_norm = params.get('use_batch_norm', True)
        self.l2_reg = params.get('l2_reg', 0.001)
        self.clip_norm = params.get('clip_norm', 1.0)
        
        # ✅ 3단계: 이제 모델 로드
        self.autoencoder = keras.models.load_model(
            model_path,
            custom_objects={'weighted_mse': EnhancedLSTMIFModel.weighted_mse}
        )
        
        # ✅ 4단계: threshold 및 기타 메타데이터 로드
        if 'threshold' not in metadata or metadata['threshold'] is None:
            raise ValueError(f"Threshold not found in metadata for {model_path.name}")
        
        self.threshold = float(metadata['threshold'])
        
        self.vacuum_range_thresholds = metadata.get('vacuum_range_thresholds', {})
        if not self.vacuum_range_thresholds:
            print(f"⚠️ Warning: No vacuum range thresholds found, using 1.0 for all ranges")
            self.vacuum_range_thresholds = {k: 1.0 for k in VacuumRangeManager.RANGES.keys()}
        
        self.model_id = metadata.get('model_id', 'unknown')
        self.created_at = metadata.get('created_at')
        self.history = metadata.get('history', {})
        self.train_stats = metadata.get('train_stats', {})
        self.val_stats = metadata.get('val_stats', {})
        self.training_time = metadata.get('training_time', 0)
        self.best_epoch = metadata.get('best_epoch', 0)
        
        self.training_params = metadata.get('training_params', {})
        
        # ✅ 5단계: Scalers 로드
        scalers_path = model_path.parent / f"{stem}_scalers.pkl"
        if not scalers_path.exists():
            raise FileNotFoundError(f"Scalers file not found: {scalers_path}")
        
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
        
        print(f"✅ Model loaded: {model_path.name}")
        print(f"   Threshold: {self.threshold:.6f}")
        print(f"   Range adjustments: {len(self.vacuum_range_thresholds)} ranges")

# ================== Isolation Forest Integration ==================
class IsolationForestDetector:
    """Isolation Forest for anomaly detection on engineered features"""
    
    def __init__(self, contamination=0.1, n_estimators=100, max_samples='auto', random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.model = None
        self.feature_scaler = StandardScaler()
        self.trained = False
        
    def extract_features(self, data):
        """
        Extract statistical and domain-specific features from time series
        Args:
            data: (n_samples, seq_len, n_features) or (seq_len, n_features)
        Returns:
            features: (n_samples, n_engineered_features)
        """
        if len(data.shape) == 2:
            data = data.reshape(1, *data.shape)
        
        n_samples = data.shape[0]
        features_list = []
        
        for i in range(n_samples):
            sample = data[i]  # (seq_len, n_features)
            
            feature_vector = []
            
            # For each sensor
            for j in range(sample.shape[1]):
                sensor_data = sample[:, j]
                
                # 🔧 NaN 체크 및 처리
                if np.all(np.isnan(sensor_data)) or len(sensor_data) == 0:
                    # 모든 값이 NaN이면 기본값 사용
                    feature_vector.extend([0.0] * 12)
                    continue
                
                # NaN 제거
                valid_data = sensor_data[~np.isnan(sensor_data)]
                if len(valid_data) == 0:
                    feature_vector.extend([0.0] * 12)
                    continue
                
                # Basic statistics
                feature_vector.extend([
                    np.mean(valid_data),
                    np.std(valid_data) if len(valid_data) > 1 else 0.0,
                    np.min(valid_data),
                    np.max(valid_data),
                    np.median(valid_data),
                    np.percentile(valid_data, 25),
                    np.percentile(valid_data, 75),
                ])
                
                # Range and IQR
                feature_vector.append(np.max(valid_data) - np.min(valid_data))
                feature_vector.append(np.percentile(valid_data, 75) - np.percentile(valid_data, 25))
                
                # Skewness and Kurtosis
                from scipy.stats import skew, kurtosis
                try:
                    feature_vector.append(skew(valid_data) if len(valid_data) > 2 else 0.0)
                    feature_vector.append(kurtosis(valid_data) if len(valid_data) > 3 else 0.0)
                except:
                    feature_vector.extend([0.0, 0.0])
                
                # Trend (linear regression slope)
                try:
                    x = np.arange(len(valid_data))
                    z = np.polyfit(x, valid_data, 1)
                    feature_vector.append(z[0])
                except:
                    feature_vector.append(0.0)
            
            # Vacuum-specific features (column 0)
            vacuum = sample[:, 0]
            vacuum = vacuum[~np.isnan(vacuum)]  # 🔧 NaN 제거
            
            if len(vacuum) == 0:
                feature_vector.extend([0.0, 0.0])
            else:
                # Vacuum recovery rate
                split_point = min(2400, len(vacuum) // 3)
                if len(vacuum) > split_point:
                    initial_vacuum = np.mean(vacuum[:split_point])
                    final_vacuum = np.mean(vacuum[split_point:])
                    vacuum_recovery = initial_vacuum - final_vacuum
                    feature_vector.append(vacuum_recovery)
                else:
                    feature_vector.append(0)
                
                # Vacuum stability
                if len(vacuum) > split_point:
                    late_vacuum_std = np.std(vacuum[split_point:])
                    feature_vector.append(late_vacuum_std)
                else:
                    feature_vector.append(np.std(vacuum) if len(vacuum) > 1 else 0.0)
            
            # Temperature correlation features
            temp_cols = [1, 2, 3, 4]
            temps = sample[:, temp_cols]
            
            # Temperature imbalance
            temp_valid = temps[~np.isnan(temps).any(axis=1)]  # 🔧 NaN 제거
            if len(temp_valid) > 0:
                temp_mean_per_pump = np.mean(temp_valid, axis=0)
                temp_imbalance = np.std(temp_mean_per_pump) if len(temp_mean_per_pump) > 1 else 0.0
                feature_vector.append(temp_imbalance)
            else:
                feature_vector.append(0.0)
            
            # Pressure features
            pressure_cols = [5, 6]
            pressures = sample[:, pressure_cols]
            
            # Pressure imbalance
            pressure_valid = pressures[~np.isnan(pressures).any(axis=1)]  # 🔧 NaN 제거
            if len(pressure_valid) > 0:
                pressure_mean = np.mean(pressure_valid, axis=0)
                pressure_imbalance = np.abs(pressure_mean[0] - pressure_mean[1]) if len(pressure_mean) == 2 else 0.0
                feature_vector.append(pressure_imbalance)
            else:
                feature_vector.append(0.0)
            
            # Cross-correlation features (vacuum vs temperature)
            vacuum_temp_corr = []
            for temp_col in temp_cols:
                try:
                    v = vacuum
                    t = sample[:, temp_col]
                    # 🔧 둘 다 유효한 인덱스만 선택
                    valid_idx = ~(np.isnan(v) | np.isnan(t))
                    if np.sum(valid_idx) > 5:  # 최소 5개 이상의 유효한 포인트
                        corr = np.corrcoef(v[valid_idx], t[valid_idx])[0, 1]
                        vacuum_temp_corr.append(corr if not np.isnan(corr) else 0.0)
                    else:
                        vacuum_temp_corr.append(0.0)
                except:
                    vacuum_temp_corr.append(0.0)
            
            feature_vector.extend(vacuum_temp_corr)
            
            # 🔧 최종 NaN 체크 및 치환
            feature_vector = [0.0 if np.isnan(x) or np.isinf(x) else x for x in feature_vector]
            
            features_list.append(feature_vector)
        
        return np.array(features_list, dtype=np.float32)  # 🔧 float32로 명시적 변환
    
    def train(self, train_data):
        """
        Train Isolation Forest on training data
        Args:
            train_data: list of (seq_len, n_features) arrays
        """
        # Extract features from all training samples
        train_features = []
        for data in train_data:
            features = self.extract_features(data)
            train_features.append(features[0])
        
        train_features = np.array(train_features, dtype=np.float32)
        
        # 🔧 NaN/Inf 체크
        if np.any(np.isnan(train_features)) or np.any(np.isinf(train_features)):
            print(f"⚠️ Warning: Found NaN/Inf in features, replacing with 0")
            train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        train_features_scaled = self.feature_scaler.fit_transform(train_features)
        
        # 🔧 Scaling 후에도 NaN 체크
        if np.any(np.isnan(train_features_scaled)) or np.any(np.isinf(train_features_scaled)):
            print(f"⚠️ Warning: Found NaN/Inf after scaling, replacing with 0")
            train_features_scaled = np.nan_to_num(train_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(train_features_scaled)
        self.trained = True
        
        return train_features_scaled
    
    def predict(self, data):
        """
        Predict anomaly score and label
        Args:
            data: (seq_len, n_features) array
        Returns:
            dict with prediction results
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        features = self.extract_features(data)
        features_scaled = self.feature_scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]  # 1: normal, -1: anomaly
        anomaly_score = self.model.score_samples(features_scaled)[0]  # More negative = more anomalous
        
        # Convert to probability-like score (0-1 range, higher = more anomalous)
        # Isolation Forest score is negative, so we negate and normalize
        anomaly_probability = 1 / (1 + np.exp(anomaly_score * 5))  # Sigmoid transformation
        
        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': float(-anomaly_score),  # Positive = more anomalous
            'anomaly_probability': float(anomaly_probability),
            'raw_score': float(anomaly_score)
        }
        
# ================== Ensemble Detector ==================
class EnsembleLSTMAE_IF:
    """
    Ensemble of LSTM-AE and Isolation Forest for robust anomaly detection
    """
    
    def __init__(self, lstmae_model, if_detector, 
                 lstmae_weight=0.7, if_weight=0.3,
                 ensemble_strategy='weighted_voting'):
        """
        Args:
            lstmae_model: Trained EnhancedLSTMIFModel instance
            if_detector: Trained IsolationForestDetector instance
            lstmae_weight: Weight for LSTM-AE predictions (0-1)
            if_weight: Weight for Isolation Forest predictions (0-1)
            ensemble_strategy: 'weighted_voting', 'max', 'min', or 'product'
        """
        self.lstmae = lstmae_model
        self.if_detector = if_detector
        self.lstmae_weight = lstmae_weight
        self.if_weight = if_weight
        self.ensemble_strategy = ensemble_strategy
        
        # Normalize weights
        total_weight = lstmae_weight + if_weight
        self.lstmae_weight /= total_weight
        self.if_weight /= total_weight
    
    def detect_anomaly(self, data, return_details=True):
        """
        Ensemble anomaly detection
        Args:
            data: (seq_len, n_features) array
            return_details: Whether to return detailed results from both models
        Returns:
            dict with ensemble prediction results
        """
        # Get predictions from both models
        lstmae_result = self.lstmae.detect_anomaly(data, return_details=True)
        if_result = self.if_detector.predict(data)
        
        # Normalize scores to 0-1 range for comparison
        # LSTM-AE: use ratio to threshold
        lstmae_normalized_score = lstmae_result['anomaly_score'] / lstmae_result['adjusted_threshold']
        
        # IF: already has probability-like score
        if_normalized_score = if_result['anomaly_probability']
        
        # Ensemble combination
        if self.ensemble_strategy == 'weighted_voting':
            ensemble_score = (self.lstmae_weight * lstmae_normalized_score + 
                            self.if_weight * if_normalized_score)
            # Threshold for ensemble: if weighted score > 1.0, it's anomaly
            is_anomaly = ensemble_score > 1.0
            
        elif self.ensemble_strategy == 'max':
            # Maximum score (pessimistic: flag as anomaly if either model strongly suggests)
            ensemble_score = max(lstmae_normalized_score, if_normalized_score)
            is_anomaly = ensemble_score > 1.0
            
        elif self.ensemble_strategy == 'min':
            # Minimum score (optimistic: both must agree)
            ensemble_score = min(lstmae_normalized_score, if_normalized_score)
            is_anomaly = ensemble_score > 1.0
            
        elif self.ensemble_strategy == 'product':
            # Geometric mean (moderate approach)
            ensemble_score = np.sqrt(lstmae_normalized_score * if_normalized_score)
            is_anomaly = ensemble_score > 1.0
        
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.ensemble_strategy}")
        
        # Calculate confidence (agreement between models)
        agreement = 1.0 - abs(lstmae_normalized_score - if_normalized_score) / 2.0
        
        result = {
            'is_anomaly': bool(is_anomaly),
            'ensemble_score': float(ensemble_score),
            'confidence': float(agreement),
            'vacuum_range': lstmae_result['vacuum_range'],
            'vacuum_value': lstmae_result['vacuum_value'],
            'ensemble_strategy': self.ensemble_strategy
        }
        
        if return_details:
            result.update({
                'lstmae_result': lstmae_result,
                'if_result': if_result,
                'lstmae_weight': self.lstmae_weight,
                'if_weight': self.if_weight,
                'lstmae_normalized_score': float(lstmae_normalized_score),
                'if_normalized_score': float(if_normalized_score)
            })
        
        return result        
                

# ================== Enhanced GUI ==================
class EnhancedLSTMAEGUI:
    def __init__(self, gpu_available=False):
        self.gpu_available = gpu_available
        
        self.root = tk.Tk()
        self.root.title("Cryo Pump AI Anomaly Detection for EV Tech Team")
        self.root.geometry("1600x950")
        
        self.base_dir = Path.cwd()
        self.training_data_dir = None
        self.stop_training_event = Event()
        
        self.model = None
        self.current_results = []
        self.loaded_models_info = {}
        self.evaluation_data = {}
        
        self.setup_gui()
    
    def setup_gui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        info_frame = ttk.LabelFrame(self.root, text="System Info", padding="5")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        gpu_text = f"GPU: {'RTX 2080Ti (FP16)' if self.gpu_available else 'CPU Mode'}"
        ttk.Label(info_frame, text=gpu_text, font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=20)
        ttk.Label(info_frame, text="Algorithm: LSTM-IF Ensemble (AE + IF)", 
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=20)
        ttk.Label(info_frame, text="All Vacuum Ranges", 
                 font=('Arial', 10, 'bold'), foreground='#2E7D32').pack(side=tk.LEFT, padx=20)
        
        # 제작자 정보 - 오른쪽 정렬
        ttk.Label(info_frame, text="Created by SeJun Kang", 
                 font=('Georgia', 9, 'italic'), foreground='#555555').pack(side=tk.RIGHT, padx=20)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.create_config_tab()
        self.create_training_tab()
        self.create_detection_tab()
        self.create_analysis_tab()
        self.create_model_manager_tab()
        self.create_evaluation_tab()
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, side=tk.BOTTOM)

    def show_architecture_guide(self):
        """Model Architecture 파라미터 가이드"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Model Architecture Guide")
        guide_window.geometry("800x700")
        
        guide_text = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD, 
                                               font=('Malgun Gothic', 10))
        guide_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
    MODEL ARCHITECTURE PARAMETERS
    ═══════════════════════════════════════════════════════════════
    
    📐 ENCODER LAYER 1 (기본값: 64, 범위: 32-128)
    ───────────────────────────────────────────────────────────────
    첫 번째 LSTM 레이어의 유닛 수
    
    ▶ 값을 증가시킬 때 (64 → 128):
      • 장점: 더 복잡한 패턴 학습 가능, 입력 데이터의 미세한 특징 포착
      • 단점: Overfitting 위험 증가, 학습 시간 및 메모리 사용량 증가
      • 권장: 데이터가 충분히 많고(>1000) 복잡할 때
    
    ▶ 값을 감소시킬 때 (64 → 32):
      • 장점: 단순한 모델로 일반화 성능 향상, 학습 속도 빠름
      • 단점: 복잡한 패턴 포착 능력 저하
      • 권장: 데이터가 적거나(<500) 단순한 패턴일 때
    
    
    📐 ENCODER LAYER 2 (기본값: 32, 범위: 16-96)
    ───────────────────────────────────────────────────────────────
    두 번째 LSTM 레이어의 유닛 수
    
    ▶ 일반 원칙:
      • Layer 1의 약 50% 수준 유지 권장 (예: Layer 1=64 → Layer 2=32)
      • 너무 급격한 감소는 정보 손실 위험
    
    ▶ 값을 증가시킬 때 (32 → 64):
      • 중간 표현(representation)의 표현력 증가
      • Layer 1에서 추출한 feature를 더 풍부하게 압축
      • 단점: 계산 비용 증가
    
    ▶ 값을 감소시킬 때 (32 → 16):
      • 더 강한 정보 압축, Noise 제거 효과 증가
      • 단점: 중요한 정보 손실 위험
    
    
    🎯 LATENT DIMENSION (기본값: 16, 범위: 8-64)
    ───────────────────────────────────────────────────────────────
    ⚠️ 이상 탐지의 핵심 파라미터!
    Bottleneck 레이어의 차원 수 - 정보 압축의 강도를 결정
    
    ▶ 값을 증가시킬 때 (16 → 32):
      • 장점: 정보 보존력 증가, Reconstruction error 감소
      • 단점: 이상 탐지 민감도 감소 (정상/비정상 구분 어려워짐)
      • 권장: 복잡한 시스템, 다양한 동작 모드가 있을 때
    
    ▶ 값을 감소시킬 때 (16 → 8):
      • 장점: 강한 압축으로 핵심 feature만 추출, 이상 탐지 민감도 증가
      • 단점: 정보 손실 가능성 증가
      • 권장: 단순한 시스템
    
    ⚠️ 주의사항:
      • 너무 크면 → 모든 것을 정상으로 판단 (이상 탐지 실패)
      • 너무 작으면 → 정상도 이상으로 판단 (과도한 오탐)
    
    
    💧 DROPOUT (기본값: 0.25, 범위: 0.0-0.5)
    ───────────────────────────────────────────────────────────────
    학습 중 무작위로 뉴런을 비활성화하여 Overfitting 방지
    
    ▶ 값을 증가시킬 때 (0.25 → 0.4):
      • 장점: Overfitting 방지 효과 강화, 일반화 성능 향상
      • 단점: 학습이 느려지고 수렴이 어려워질 수 있음
      • 권장: 데이터가 적거나(<300) overfitting 징후가 보일 때
    
    ▶ 값을 감소시킬 때 (0.25 → 0.1):
      • 장점: 학습 속도 향상, Training data를 더 잘 학습
      • 단점: Overfitting 위험 증가
      • 권장: 데이터가 충분히 많을 때(>1000)
    
    ▶ 0.0으로 설정:
      • Dropout 비활성화
      • 권장: 데이터가 매우 많고 다양할 때만 사용
    
    
    🔁 RECURRENT DROPOUT (기본값: 0.15, 범위: 0.0-0.3)
    ───────────────────────────────────────────────────────────────
    LSTM의 순환 연결(recurrent connection)에 적용되는 Dropout
    
    ▶ 일반 Dropout과의 차이:
      • 일반 Dropout: 입력-출력 연결에 적용
      • Recurrent Dropout: 시간 축의 연결에 적용 (시계열 overfitting 방지)
    
    ▶ 값을 증가시킬 때 (0.15 → 0.25):
      • 장점: 시간적 의존성 학습이 더 견고해짐
      • 특히 효과적: Sequence 길이가 길 때
      • 단점: 학습 시간 증가
    
    ▶ 값을 감소시킬 때 (0.15 → 0.05):
      • 장점: 시간적 패턴 학습 능력 향상
      • 권장: 짧은 sequence 데이터
    
    
    ⚖️ L2 REGULARIZATION (기본값: 0.001, 범위: 0.0-0.01)
    ───────────────────────────────────────────────────────────────
    가중치의 크기에 패널티를 부여하여 Overfitting 방지
    
    ▶ 값을 증가시킬 때 (0.001 → 0.005):
      • 가중치가 작게 유지되어 모델이 단순해짐
      • Overfitting 방지 효과 증가
      • 권장: 복잡한 모델, 데이터 적을 때
    
    ▶ 값을 감소시킬 때 (0.001 → 0.0001):
      • 모델의 표현력 증가
      • 권장: 데이터가 많고 복잡할 때
    
    
    ✂️ GRADIENT CLIP NORM (기본값: 1.0, 범위: 0.1-5.0)
    ───────────────────────────────────────────────────────────────
    Gradient의 최대 크기를 제한하여 학습 안정화
    
    ▶ 값을 증가시킬 때 (1.0 → 2.0):
      • 더 큰 gradient 허용 → 빠른 학습 가능
      • 단점: 학습 불안정 위험
    
    ▶ 값을 감소시킬 때 (1.0 → 0.5):
      • 더 안정적인 학습
      • 권장: 학습이 불안정할 때
    
    
    🔀 BIDIRECTIONAL (기본값: True)
    ───────────────────────────────────────────────────────────────
    양방향 LSTM 사용 여부
    
    ▶ True (양방향):
      • 과거와 미래 정보를 모두 활용
      • 장점: 패턴 인식 성능 향상
      • 단점: 계산량 약 2배 증가, 메모리 사용량 증가
      • 권장: 대부분의 경우 (정확도 우선)
    
    ▶ False (단방향):
      • 과거 정보만 활용 (순방향)
      • 장점: 계산 속도 빠름, 메모리 효율적, 실시간 처리 유리
      • 단점: 성능은 다소 낮을 수 있음
      • 권장: 빠른 학습/추론이 필요할 때
    
    
    ═══════════════════════════════════════════════════════════════
    💡 조정 팁
    ═══════════════════════════════════════════════════════════════
    
    ▶ Overfitting 발생 시:
      • Dropout ↑, Recurrent Dropout ↑
      • Encoder units ↓, Latent dimension ↓
      • L2 Regularization ↑
    
    ▶ Underfitting 발생 시:
      • Encoder units ↑, Latent dimension ↑
      • Dropout ↓
    
    ▶ 학습 불안정 시:
      • Gradient Clip Norm ↓
      • Learning Rate ↓ (Training Parameters에서)
    
    ▶ 데이터 양에 따른 권장 설정:
      • 데이터 적음(<300): Encoder=[32,16], Latent=8, Dropout=0.3-0.4
      • 데이터 보통(300-1000): 기본값 사용
      • 데이터 많음(>1000): Encoder=[96,48], Latent=32, Dropout=0.15-0.2
        """
        
        guide_text.insert('1.0', content)
        guide_text.config(state='disabled')
        
        ttk.Button(guide_window, text="닫기", command=guide_window.destroy).pack(pady=10)
    
    
    def show_training_guide(self):
        """Training Parameters 가이드"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Training Parameters Guide")
        guide_window.geometry("800x700")
        
        guide_text = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD, 
                                               font=('Malgun Gothic', 10))
        guide_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
    TRAINING PARAMETERS
    ═══════════════════════════════════════════════════════════════
    
    🔄 EPOCHS (기본값: 100, 범위: 50-300)
    ───────────────────────────────────────────────────────────────
    전체 데이터셋을 학습하는 반복 횟수
    
    ▶ 값을 증가시킬 때 (100 → 200):
      • 장점: 모델이 데이터를 더 오래 학습, 수렴 가능성 증가
      • 단점: Overfitting 위험 증가, 학습 시간 증가
      • 💡 팁: Early stopping(patience)이 있으므로 여유있게 설정 권장
    
    ▶ 값을 감소시킬 때 (100 → 50):
      • 장점: 학습 시간 단축
      • 단점: Underfitting 위험 (학습 부족)
      • 권장: 빠른 실험용
    
    ▶ 권장 설정:
      • 100-150으로 설정하고 Patience로 실제 학습 길이 조절
    
    
    📦 BATCH SIZE (기본값: 32, 범위: 8-128)
    ───────────────────────────────────────────────────────────────
    한 번에 처리하는 샘플 수
    
    ▶ 값을 증가시킬 때 (32 → 64):
      • 장점: GPU 활용도 증가 → 학습 속도 향상, Gradient 추정 안정적
      • 단점: 메모리 사용량 증가, 일반화 성능 다소 저하 가능
      • 주의: Learning rate 조정 필요할 수 있음
    
    ▶ 값을 감소시킬 때 (32 → 16):
      • 장점: 메모리 사용량 감소, Gradient에 noise 증가 → Regularization 효과
      • 단점: 학습 불안정, Epoch당 업데이트 횟수 증가 → 느림
      • 권장: 메모리 부족하거나 일반화 성능 향상 원할 때
    
    ▶ RTX 2080Ti 최적값: 32-64
    
    
    📚 LEARNING RATE (기본값: 0.001)
    ───────────────────────────────────────────────────────────────
    모델 가중치 업데이트 속도 (학습률)
    
    ▶ 값을 증가시킬 때 (0.001 → 0.005):
      • 장점: 빠른 학습
      • 단점: 최적점을 지나칠 위험(overshooting), 학습 불안정
      • 권장: 빠른 수렴이 필요할 때 시도
    
    ▶ 값을 감소시킬 때 (0.001 → 0.0001):
      • 장점: 안정적인 학습, 정밀한 최적화
      • 단점: 학습 시간 크게 증가, 학습 정체 가능
      • 권장: Fine-tuning 단계
    
    ▶ 권장 전략:
      • 초기: 0.001
      • LR Schedule 사용하여 자동 조정 (plateau 권장)
    
    
    ✂️ VALIDATION SPLIT (기본값: 0.2, 범위: 0.1-0.4)
    ───────────────────────────────────────────────────────────────
    검증용 데이터 비율
    
    ▶ 값을 증가시킬 때 (0.2 → 0.3):
      • 장점: Validation set 커짐 → 성능 평가 신뢰도 증가
      • 단점: Training 데이터 감소
      • 권장: 데이터가 충분히 많을 때만 (>1000)
    
    ▶ 값을 감소시킬 때 (0.2 → 0.1):
      • 장점: Training 데이터 증가
      • 단점: Validation 신뢰도 감소
      • 권장: 데이터가 적을 때 (<500)
    
    ▶ 권장 설정:
      • 데이터 <500개: 0.15-0.2
      • 데이터 >1000개: 0.2-0.3
    
    
    ⏸️ PATIENCE (기본값: 20, 범위: 10-50)
    ───────────────────────────────────────────────────────────────
    성능 개선 없을 때 대기할 Epoch 수 (Early Stopping)
    
    ▶ 값을 증가시킬 때 (20 → 40):
      • 장점: 느리게 개선되는 경우를 놓치지 않음
      • 단점: 불필요하게 긴 학습 시간
      • 권장: 복잡한 모델이나 어려운 데이터일 때
    
    ▶ 값을 감소시킬 때 (20 → 10):
      • 장점: 빠른 조기 종료 → 시간 절약
      • 단점: 최적점 도달 전 종료 위험
      • 권장: 빠른 실험용
    
    ▶ 권장 설정: Epochs의 15-20% 수준
    
    
    🎯 MIN DELTA (기본값: 0.0001)
    ───────────────────────────────────────────────────────────────
    의미있는 개선으로 간주할 최소 변화량
    
    ▶ 값을 증가시킬 때 (0.0001 → 0.001):
      • 더 큰 개선만 의미있다고 판단 → 조기 종료 가능성 증가
      • 미세한 개선 무시
      • 권장: 빠른 실험
    
    ▶ 값을 감소시킬 때 (0.0001 → 0.00001):
      • 아주 작은 개선도 의미있다고 판단 → 학습 더 오래 진행
      • 단점: Overfitting 위험 증가
      • 권장: 정밀한 최적화 필요시
    
    
    🎨 AUGMENTATION (기본값: 3, 범위: 0-5)
    ───────────────────────────────────────────────────────────────
    데이터 증강 배수 (원본 데이터에 noise 추가하여 확장)
    
    ▶ 작동 방식:
      • Augmentation=3 → 원본 1개당 추가 6개 생성 (총 7배)
      • Noise, Scaling 등을 적용하여 데이터 다양성 증가
    
    ▶ 값을 증가시킬 때 (3 → 5):
      • 장점: Training 데이터 대폭 증가, Overfitting 방지 강화, Robustness 향상
      • 단점: 학습 시간 크게 증가, 메모리 사용량 증가
      • 권장: 데이터가 적을 때 (<300) 매우 효과적
    
    ▶ 값을 감소시킬 때 (3 → 1):
      • 장점: 학습 시간 단축, 메모리 효율
      • 단점: Overfitting 위험 증가
      • 권장: 데이터가 충분히 많을 때 (>1000)
    
    ▶ 0으로 설정:
      • 데이터 증강 없음
      • 권장: 데이터가 매우 많고 다양할 때만
    
    
    ═══════════════════════════════════════════════════════════════
    💡 상황별 조정 가이드
    ═══════════════════════════════════════════════════════════════
    
    ▶ 데이터 양에 따른 설정:
    
      데이터 적음 (<300):
        • Augmentation: 4-5 ↑
        • Validation Split: 0.15 ↓
        • Patience: 25-30 ↑
    
      데이터 보통 (300-1000):
        • 기본값 사용
    
      데이터 많음 (>1000):
        • Augmentation: 1-2 ↓
        • Validation Split: 0.25-0.3 ↑
        • Batch Size: 64 ↑
    
    
    ▶ 학습 속도가 느릴 때:
      • Batch Size ↑
      • Augmentation ↓
      • Epochs ↓
    
    
    ▶ 학습이 불안정할 때:
      • Batch Size ↑
      • Learning Rate ↓
      • Patience ↑
    
    
    ▶ Overfitting 발생 시:
      • Augmentation ↑
      • Validation Split ↑ (데이터 많을 때만)
      • Dropout, Recurrent Dropout ↑ (Architecture에서)
        """
        
        guide_text.insert('1.0', content)
        guide_text.config(state='disabled')
        
        ttk.Button(guide_window, text="닫기", command=guide_window.destroy).pack(pady=10)
    
    
    def show_advanced_guide(self):
        """Advanced Options 가이드"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Advanced Options Guide")
        guide_window.geometry("800x700")
        
        guide_text = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD, 
                                               font=('Malgun Gothic', 10))
        guide_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
    ADVANCED OPTIONS
    ═══════════════════════════════════════════════════════════════
    
    🔧 OPTIMIZER (기본값: adam)
    ───────────────────────────────────────────────────────────────
    가중치 업데이트 알고리즘 선택
    
    ▶ adam (Adaptive Moment Estimation) - 기본 권장:
      • 가장 범용적이고 안정적
      • Learning rate를 자동으로 조정
      • 대부분의 경우 최선의 선택
      • 장점: 빠른 수렴, 안정적, 하이퍼파라미터 튜닝 덜 필요
      • 단점: 메모리 사용량 다소 높음
    
    ▶ sgd (Stochastic Gradient Descent):
      • 가장 기본적인 최적화 알고리즘
      • Momentum과 Nesterov 가속 적용됨
      • 장점: 메모리 효율적, 최종 성능이 더 좋을 수 있음
      • 단점: 학습 느림, Learning rate 튜닝 중요
      • 권장: Fine-tuning 단계, 정밀한 최적화 필요시
    
    ▶ rmsprop (Root Mean Square Propagation):
      • RNN/LSTM에 효과적
      • Learning rate를 자동 조정하지만 Adam보다 단순
      • 장점: Adam보다 메모리 효율적
      • 단점: Adam보다 성능 다소 낮을 수 있음
      • 권장: 메모리 제약이 있을 때
    
    ▶ adamax (Adam의 변형):
      • Adam의 무한대 norm 버전
      • 장점: 큰 gradient에 더 robust
      • 단점: Adam과 큰 차이 없는 경우 많음
      • 권장: Adam 성능 불만족 시 시도
    
    
    🎭 LATENT ACTIVATION (기본값: tanh)
    ───────────────────────────────────────────────────────────────
    Latent space의 활성화 함수
    
    ▶ tanh (Hyperbolic Tangent) - 기본 권장:
      • 출력 범위: -1 ~ 1
      • 장점: 중심이 0, Gradient 안정적, Autoencoder에 전통적으로 사용
      • 권장: 대부분의 경우
    
    ▶ relu (Rectified Linear Unit):
      • 출력 범위: 0 ~ ∞
      • 장점: 학습 빠름, Gradient vanishing 문제 없음
      • 단점: Dead neuron 문제 가능
      • 권장: 빠른 학습 필요시
    
    ▶ sigmoid:
      • 출력 범위: 0 ~ 1
      • 장점: 확률적 해석 가능
      • 단점: Gradient vanishing 문제, 학습 느림
      • 권장: 특별한 경우 외 비추천
    
    ▶ linear:
      • 활성화 함수 없음 (y = x)
      • 장점: 정보 손실 없음
      • 단점: 비선형성 없음 → 표현력 저하
      • 권장: 특수한 실험용
    
    ▶ elu (Exponential Linear Unit):
      • ReLU의 개선 버전
      • 음수 값 허용, Gradient 더 smooth
      • 권장: ReLU 대안으로 시도
    
    ▶ selu (Scaled ELU):
      • Self-normalizing 특성
      • 특정 조건에서 자동으로 정규화
      • 권장: 깊은 네트워크에서 실험
    
    
    🧮 BATCH NORMALIZATION (기본값: True)
    ───────────────────────────────────────────────────────────────
    각 레이어의 출력을 정규화하여 학습 안정화
    
    ▶ True (활성화) - 기본 권장:
      • 장점: 학습 안정화, 빠른 수렴, 높은 Learning rate 사용 가능
      • 장점: Internal Covariate Shift 방지
      • 장점: Regularization 효과 (약간의 Dropout 역할)
      • 단점: 계산 비용 소폭 증가, 배치 크기에 의존적
    
    ▶ False (비활성화):
      • 장점: 계산 속도 소폭 향상, 배치 크기 독립적
      • 단점: 학습 불안정 가능, 더 신중한 하이퍼파라미터 튜닝 필요
      • 권장: 특별한 이유 없으면 비추천
    
    
    📅 LR SCHEDULE (기본값: plateau)
    ───────────────────────────────────────────────────────────────
    Learning rate 자동 조정 전략
    
    ▶ plateau - 기본 권장:
      • Validation loss가 개선되지 않으면 LR 감소
      • 장점: 가장 안정적, 범용적, 자동으로 최적 시점 판단
      • 장점: 데이터 특성 자동 반영
      • 권장: 대부분의 경우 최선
    
    ▶ exponential:
      • Epoch마다 고정 비율로 감소 (LR × 0.95^epoch)
      • 장점: 예측 가능한 감소 패턴
      • 단점: 데이터 특성 무시, 초반 너무 빨리 감소 가능
      • 권장: 학습 패턴 예측 가능할 때
    
    ▶ cosine:
      • Cosine 곡선 형태로 감소
      • 초반 빠르게, 후반 천천히 감소
      • 장점: 학습 말기 미세 조정 유리
      • 단점: 전체 epoch 수를 정확히 알아야 함
      • 권장: Epoch 수가 고정되어 있을 때
    
    ▶ none:
      • Learning rate 고정
      • 수동 제어 원할 때
      • 일반적으로 비추천
    
    
    💾 SAVE CHECKPOINTS (기본값: False)
    ───────────────────────────────────────────────────────────────
    학습 중 모델 체크포인트 저장
    
    ▶ True:
      • 각 성능 개선 시점마다 모델 저장
      • 장점: 최적 모델 회복 가능, 학습 과정 분석 가능
      • 장점: 학습 중단 시 재개 가능
      • 단점: 디스크 공간 많이 사용 (모델당 ~100-500MB)
      • 권장: 긴 학습(>1시간), 중요한 실험
    
    ▶ False:
      • 최종 모델만 저장
      • 장점: 디스크 효율적
      • 권장: 일반적인 학습, 빠른 실험
    
    
    ═══════════════════════════════════════════════════════════════
    💡 조합 추천
    ═══════════════════════════════════════════════════════════════
    
    ▶ 기본 설정 (대부분의 경우):
      • Optimizer: adam
      • Latent Activation: tanh
      • Batch Normalization: True
      • LR Schedule: plateau
      • Save Checkpoints: False
    
    ▶ 빠른 실험용:
      • Optimizer: adam
      • Batch Normalization: True
      • LR Schedule: none
      • Save Checkpoints: False
    
    ▶ 정밀한 최적화:
      • Optimizer: adam → sgd (fine-tuning 단계)
      • Latent Activation: tanh
      • Batch Normalization: True
      • LR Schedule: cosine
      • Save Checkpoints: True
    
    ▶ 메모리 제약 환경:
      • Optimizer: rmsprop
      • Batch Normalization: False
      • LR Schedule: plateau
      • Save Checkpoints: False
    
    ▶ 학습 불안정 시:
      • Optimizer: adam
      • Batch Normalization: True ✓
      • LR Schedule: plateau ✓
      • Learning Rate ↓ (Training Parameters에서)
        """
        
        guide_text.insert('1.0', content)
        guide_text.config(state='disabled')
        
        ttk.Button(guide_window, text="닫기", command=guide_window.destroy).pack(pady=10)
    
    
    def show_ensemble_guide(self):
        """Ensemble Configuration 가이드"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Ensemble Configuration Guide")
        guide_window.geometry("800x700")
        
        guide_text = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD, 
                                               font=('Malgun Gothic', 10))
        guide_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = """
    ENSEMBLE CONFIGURATION (LSTM-AE + Isolation Forest)
    ═══════════════════════════════════════════════════════════════
    
    🔥 ENABLE ENSEMBLE (기본값: True)
    ───────────────────────────────────────────────────────────────
    Isolation Forest를 추가하여 앙상블 탐지 활성화
    
    ▶ 앙상블의 원리:
      • LSTM-AE: 시계열 패턴 기반 탐지 (시간적 이상)
      • Isolation Forest: 통계적 특징 기반 탐지 (공간적 이상)
      • 두 방법을 결합하여 더 강건하고 정확한 탐지
    
    ▶ True (활성화) - 권장:
      • 장점: 탐지 정확도 향상, False Positive/Negative 감소
      • 장점: 두 모델의 장점 결합 → 더 robust
      • 단점: 학습 시간 1-3분 추가, 메모리 약간 증가
      • 권장: 실제 운영 환경, 정확도 중요시
    
    ▶ False (비활성화):
      • LSTM-AE만 사용
      • 장점: 빠른 학습, 단순한 파이프라인
      • 권장: 빠른 실험, 리소스 제약
    
    
    🌲 IF CONTAMINATION (기본값: 0.05, 범위: 0.01-0.5)
    ───────────────────────────────────────────────────────────────
    Isolation Forest의 예상 이상치 비율
    
    ▶ 의미:
      • Training data 중 이상치로 예상되는 비율
      • IF가 이 비율만큼을 이상으로 분류하려 함
    
    ▶ 값을 감소시킬 때 (0.05 → 0.02):
      • IF가 더 보수적으로 판단 (이상 판정 줄어듦)
      • 장점: False Positive 감소
      • 단점: False Negative 증가 가능
      • 권장: Training data가 깨끗할 때
    
    ▶ 값을 증가시킬 때 (0.05 → 0.1):
      • IF가 더 공격적으로 판단 (이상 판정 늘어남)
      • 장점: 더 많은 이상 포착
      • 단점: False Positive 증가 가능
      • 권장: Training data에 이상치가 섞여 있을 가능성 있을 때
    
    ▶ 권장 설정:
      • 일반적: 0.05 (5%)
      • 깨끗한 데이터: 0.01-0.03
      • 불확실한 데이터: 0.05-0.1
    
    
    🌳 IF N ESTIMATORS (기본값: 100, 범위: 50-500)
    ───────────────────────────────────────────────────────────────
    Isolation Forest의 트리 개수
    
    ▶ 값을 증가시킬 때 (100 → 200):
      • 장점: 더 안정적이고 정확한 예측
      • 단점: 학습 시간 증가 (선형 비례)
      • 권장: 데이터가 복잡하거나 정확도 중요시
    
    ▶ 값을 감소시킬 때 (100 → 50):
      • 장점: 빠른 학습
      • 단점: 예측 불안정, 정확도 저하 가능
      • 권장: 빠른 실험용
    
    ▶ 권장 설정:
      • 빠른 실험: 50-75
      • 일반적: 100-150
      • 높은 정확도: 200-300
      • 과도(비추천): 500+ (시간 대비 효과 미미)
    
    
    🎯 ENSEMBLE STRATEGY (기본값: weighted_voting)
    ───────────────────────────────────────────────────────────────
    LSTM-AE와 IF의 결과를 결합하는 방법
    
    ▶ weighted_voting - 기본 권장:
      • 각 모델의 점수에 가중치를 곱하여 합산
      • Score = (LSTM-AE_score × w1) + (IF_score × w2)
      • 장점: 가장 균형잡힌 결과, 가중치로 미세 조정 가능
      • 권장: 대부분의 경우
    
    ▶ max:
      • 두 모델 중 높은 점수 선택 (비관적 접근)
      • 하나라도 강하게 의심하면 이상 판정
      • 장점: 이상 놓칠 위험 최소화
      • 단점: False Positive 증가 가능
      • 권장: 이상 탐지 놓치면 안될 때 (안전 중요)
    
    ▶ min:
      • 두 모델 중 낮은 점수 선택 (낙관적 접근)
      • 둘 다 의심해야 이상 판정
      • 장점: False Positive 최소화
      • 단점: 이상 놓칠 위험 증가
      • 권장: False alarm 최소화 중요시
    
    ▶ product:
      • 기하평균 방식
      • Score = √(LSTM-AE_score × IF_score)
      • 장점: 중간 정도의 균형
      • 단점: weighted_voting과 큰 차이 없는 경우 많음
      • 권장: weighted_voting 대안으로 실험
    
    
    ⚖️ LSTMAE WEIGHT (기본값: 0.6, 범위: 0.0-1.0)
    ───────────────────────────────────────────────────────────────
    Ensemble에서 LSTM-AE의 가중치 (weighted_voting 전략 사용 시)
    
    ▶ 값을 증가시킬 때 (0.6 → 0.8):
      • LSTM-AE의 영향력 증가
      • 시계열 패턴 기반 탐지 강조
      • 권장: 시간적 이상 패턴이 중요할 때
    
    ▶ 값을 감소시킬 때 (0.6 → 0.4):
      • IF의 영향력 상대적 증가
      • 통계적 특징 기반 탐지 강조
      • 권장: 순간적 이상값 탐지가 중요할 때
    
    ▶ 권장 설정:
      • 시계열 중요: 0.7-0.8
      • 균형: 0.5-0.6 (기본)
      • 통계적 특징 중요: 0.3-0.4
    
    
    ⚖️ IF WEIGHT (기본값: 0.4, 범위: 0.0-1.0)
    ───────────────────────────────────────────────────────────────
    Ensemble에서 Isolation Forest의 가중치
    
    ▶ LSTMAE Weight와 합이 자동 정규화됨:
      • 실제 사용 시 LSTMAE_W와 IF_W의 합으로 나누어짐
      • 예: (0.6, 0.4) → 실제로는 (0.6/1.0, 0.4/1.0) = (0.6, 0.4)
    
    ▶ 값 조정:
      • LSTMAE Weight와 반대 방향으로 조정
      • 두 값의 비율이 중요 (절대값보다)
    
    
    ═══════════════════════════════════════════════════════════════
    💡 상황별 추천 설정
    ═══════════════════════════════════════════════════════════════
    
    ▶ 균형잡힌 기본 설정 (권장):
      • Enable Ensemble: True
      • IF Contamination: 0.05
      • IF N Estimators: 100
      • Strategy: weighted_voting
      • LSTMAE Weight: 0.6, IF Weight: 0.4
    
    ▶ 빠른 실험:
      • Enable Ensemble: False
      • (또는 IF N Estimators: 50)
    
    ▶ 최대 정확도 (시간 여유 있을 때):
      • Enable Ensemble: True
      • IF N Estimators: 200
      • Strategy: weighted_voting
      • LSTMAE Weight: 0.6, IF Weight: 0.4
    
    ▶ 안전 중요 (이상 놓치면 안됨):
      • Enable Ensemble: True
      • IF Contamination: 0.08-0.1
      • Strategy: max
      • LSTMAE Weight: 0.7, IF Weight: 0.3
    
    ▶ False alarm 최소화:
      • Enable Ensemble: True
      • IF Contamination: 0.02-0.03
      • Strategy: min
      • LSTMAE Weight: 0.5, IF Weight: 0.5
    
    ▶ 시계열 패턴 중요:
      • Strategy: weighted_voting
      • LSTMAE Weight: 0.7-0.8
      • IF Weight: 0.2-0.3
    
    ▶ 통계적 이상 중요:
      • Strategy: weighted_voting
      • LSTMAE Weight: 0.3-0.4
      • IF Weight: 0.6-0.7
    
    
    ═══════════════════════════════════════════════════════════════
    🔬 앙상블 효과 이해하기
    ═══════════════════════════════════════════════════════════════
    
    LSTM-AE가 잘 잡는 이상:
      • 시간에 따른 점진적 변화
      • 주기적 패턴 이탈
      • 순서/흐름의 이상
    
    IF가 잘 잡는 이상:
      • 순간적인 급격한 변화
      • 통계적으로 희귀한 값
      • 센서 간 관계 이상
    
    앙상블이 특히 효과적인 경우:
      • 두 가지 유형의 이상이 모두 중요할 때
      • False Positive/Negative 최소화가 중요할 때
      • 다양한 형태의 이상을 포괄적으로 탐지해야 할 때
        """
        
        guide_text.insert('1.0', content)
        guide_text.config(state='disabled')
        
        ttk.Button(guide_window, text="닫기", command=guide_window.destroy).pack(pady=10)
    
    def create_config_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚙ Configuration")
        
        # ========== 상단 3개 섹션 (좌/중/우) ==========
        top_container = ttk.Frame(frame)
        top_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 좌측: Model Architecture
        arch_frame = ttk.LabelFrame(top_container, text="🏗️ Model Architecture", padding="10")
        arch_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        self.param_vars = {}
        
        arch_params = [
            ('Encoder Layer 1', 'enc0', 64, 32, 128, 16),
            ('Encoder Layer 2', 'enc1', 32, 16, 96, 16),
            ('Latent Dimension', 'latent', 16, 8, 64, 8),
            ('Dropout', 'dropout', 0.25, 0.0, 0.5, 0.05),
            ('Recurrent Dropout', 'rec_drop', 0.15, 0.0, 0.3, 0.05),
            ('L2 Regularization', 'l2_reg', 0.001, 0.0, 0.01, 0.001),
            ('Gradient Clip Norm', 'clip_norm', 1.0, 0.1, 5.0, 0.1),
        ]
        
        for idx, (label, key, default, min_v, max_v, step) in enumerate(arch_params):
            ttk.Label(arch_frame, text=label).grid(row=idx, column=0, padx=5, pady=3, sticky=tk.W)
            if 'dropout' in key.lower() or 'l2' in key.lower() or 'clip' in key.lower():
                var = tk.DoubleVar(value=default)
            else:
                var = tk.IntVar(value=default)
            ttk.Spinbox(arch_frame, from_=min_v, to=max_v, increment=step, 
                       textvariable=var, width=12).grid(row=idx, column=1, padx=5, pady=3)
            self.param_vars[key] = var
        
        # Bidirectional
        ttk.Label(arch_frame, text="Bidirectional").grid(row=len(arch_params), column=0, padx=5, pady=3, sticky=tk.W)
        self.param_vars['bidir'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(arch_frame, variable=self.param_vars['bidir']).grid(
            row=len(arch_params), column=1, padx=5, pady=3, sticky=tk.W)
        
        # ✅ Architecture Guide 버튼 추가
        ttk.Button(arch_frame, text="📖 Architecture Guide", 
                  command=self.show_architecture_guide).grid(
                      row=len(arch_params)+1, column=0, columnspan=2, padx=5, pady=10, sticky='ew')
        
        # 중앙: Training Parameters
        train_frame = ttk.LabelFrame(top_container, text="🎓 Training Parameters", padding="10")
        train_frame.grid(row=0, column=1, sticky='nsew', padx=5)
        
        train_params = [
            ('Epochs', 'epochs', 100, 50, 300, 10),
            ('Batch Size', 'batch', 32, 8, 128, 8),
            ('Learning Rate', 'lr', 0.001, None, None, None),
            ('Validation Split', 'val_split', 0.2, 0.1, 0.4, 0.05),
            ('Patience', 'patience', 20, 10, 50, 5),
            ('Min Delta', 'min_delta', 0.0001, None, None, None),
            ('Augmentation', 'augment', 3, 0, 5, 1)
        ]
        
        for idx, param in enumerate(train_params):
            label, key, default = param[:3]
            
            ttk.Label(train_frame, text=f"{label}:").grid(row=idx, column=0, padx=5, pady=3, sticky=tk.W)
            
            if key in ['lr', 'min_delta']:
                var = tk.DoubleVar(value=default)
                ttk.Entry(train_frame, textvariable=var, width=12).grid(row=idx, column=1, padx=5, pady=3)
            elif key == 'val_split':
                var = tk.DoubleVar(value=default)
                ttk.Spinbox(train_frame, from_=param[3], to=param[4], increment=param[5],
                           textvariable=var, width=12).grid(row=idx, column=1, padx=5, pady=3)
            else:
                var = tk.IntVar(value=default)
                ttk.Spinbox(train_frame, from_=param[3], to=param[4], increment=param[5],
                           textvariable=var, width=12).grid(row=idx, column=1, padx=5, pady=3)
            
            self.param_vars[key] = var
        
        # ✅ Training Guide 버튼 추가
        ttk.Button(train_frame, text="📖 Training Guide", 
                  command=self.show_training_guide).grid(
                      row=len(train_params), column=0, columnspan=2, padx=5, pady=10, sticky='ew')
        
        # 우측: Advanced Options
        adv_frame = ttk.LabelFrame(top_container, text="⚙️ Advanced Options", padding="10")
        adv_frame.grid(row=0, column=2, sticky='nsew', padx=(5, 0))
        
        # Optimizer 선택
        ttk.Label(adv_frame, text="Optimizer:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.param_vars['optimizer'] = tk.StringVar(value='adam')
        ttk.Combobox(adv_frame, textvariable=self.param_vars['optimizer'],
                    values=['adam', 'sgd', 'rmsprop', 'adamax'],
                    width=15, state='readonly').grid(row=0, column=1, padx=5, pady=3)
        
        # Activation Function 선택
        ttk.Label(adv_frame, text="Latent Activation:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.param_vars['latent_activation'] = tk.StringVar(value='tanh')
        ttk.Combobox(adv_frame, textvariable=self.param_vars['latent_activation'],
                    values=['tanh', 'relu', 'sigmoid', 'linear', 'elu', 'selu'],
                    width=15, state='readonly').grid(row=1, column=1, padx=5, pady=3)
        
        # Batch Normalization 선택
        self.param_vars['use_batch_norm'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(adv_frame, text="Use Batch Norm",
                       variable=self.param_vars['use_batch_norm']).grid(
                           row=2, column=0, columnspan=2, padx=5, pady=3, sticky=tk.W)
        
        # LR Schedule
        ttk.Label(adv_frame, text="LR Schedule:").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W)
        self.param_vars['lr_schedule'] = tk.StringVar(value='plateau')
        ttk.Combobox(adv_frame, textvariable=self.param_vars['lr_schedule'],
                    values=['plateau', 'exponential', 'cosine', 'none'],
                    width=15, state='readonly').grid(row=3, column=1, padx=5, pady=3)
        
        # Save Checkpoints
        self.param_vars['save_checkpoints'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv_frame, text="Save Checkpoints",
                       variable=self.param_vars['save_checkpoints']).grid(
                           row=4, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # ✅ Advanced Guide 버튼 추가 (기존 Config Guide 제거하고 대체)
        ttk.Button(adv_frame, text="📖 Advanced Guide", 
                  command=self.show_advanced_guide).grid(
                      row=5, column=0, columnspan=2, padx=5, pady=10, sticky='ew')
        
        # Grid weights for equal width distribution
        top_container.grid_columnconfigure(0, weight=1, uniform='cols')
        top_container.grid_columnconfigure(1, weight=1, uniform='cols')
        top_container.grid_columnconfigure(2, weight=1, uniform='cols')
        top_container.grid_rowconfigure(0, weight=1)
        
        # ========== 하단: Ensemble Configuration ==========
        ensemble_frame = ttk.LabelFrame(frame, text="🔥 Ensemble Configuration (LSTM-AE + Isolation Forest)", padding="10")
        ensemble_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Ensemble 설정을 2열로 구성
        ensemble_left = ttk.Frame(ensemble_frame)
        ensemble_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        ensemble_right = ttk.Frame(ensemble_frame)
        ensemble_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 좌측 컬럼
        self.param_vars['enable_ensemble'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(ensemble_left, text="Enable Isolation Forest Ensemble",
                       variable=self.param_vars['enable_ensemble']).grid(
                           row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(ensemble_left, text="IF Contamination:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.param_vars['if_contamination'] = tk.DoubleVar(value=0.05)
        ttk.Spinbox(ensemble_left, from_=0.01, to=0.5, increment=0.01,
                   textvariable=self.param_vars['if_contamination'], width=12).grid(row=1, column=1, padx=5, pady=3)
        
        ttk.Label(ensemble_left, text="IF N Estimators:").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        self.param_vars['if_n_estimators'] = tk.IntVar(value=100)
        ttk.Spinbox(ensemble_left, from_=50, to=500, increment=50,
                   textvariable=self.param_vars['if_n_estimators'], width=12).grid(row=2, column=1, padx=5, pady=3)
        
        ttk.Label(ensemble_left, text="Ensemble Strategy:").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W)
        self.param_vars['ensemble_strategy'] = tk.StringVar(value='weighted_voting')
        ttk.Combobox(ensemble_left, textvariable=self.param_vars['ensemble_strategy'],
                    values=['weighted_voting', 'max', 'min', 'product'],
                    width=15, state='readonly').grid(row=3, column=1, padx=5, pady=3)
        
        # 우측 컬럼
        ttk.Label(ensemble_right, text="LSTM-AE Weight:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.param_vars['lstmae_weight'] = tk.DoubleVar(value=0.6)
        ttk.Spinbox(ensemble_right, from_=0.0, to=1.0, increment=0.1,
                   textvariable=self.param_vars['lstmae_weight'], width=12).grid(row=0, column=1, padx=5, pady=3)
        
        ttk.Label(ensemble_right, text="IF Weight:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.param_vars['if_weight'] = tk.DoubleVar(value=0.4)
        ttk.Spinbox(ensemble_right, from_=0.0, to=1.0, increment=0.1,
                   textvariable=self.param_vars['if_weight'], width=12).grid(row=1, column=1, padx=5, pady=3)
        
        # ✅ Ensemble Guide 버튼 추가
        ttk.Button(ensemble_right, text="📖 Ensemble Guide", 
                  command=self.show_ensemble_guide).grid(
                      row=2, column=0, columnspan=2, padx=5, pady=10, sticky='ew')
        
        # Info label (하단 전체 너비) - 기존 유지
        info_text = ("💡 Ensemble combines LSTM-AE (temporal patterns) + IF (feature-based)\n"
                     "   Recommended: weighted_voting with 0.6/0.4 weights for balanced detection")
        ttk.Label(ensemble_frame, text=info_text, foreground='gray', 
                  font=('Arial', 8), justify=tk.LEFT).pack(pady=(10, 0))
    
    def create_training_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🎓 Training")
        
        data_frame = ttk.LabelFrame(frame, text="Training Data", padding="10")
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(data_frame, text="Data Folder:").grid(row=0, column=0, sticky=tk.W)
        self.train_path_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.train_path_var, width=70).grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="Browse", command=self.select_train_folder).grid(row=0, column=2)
        
        ttk.Label(data_frame, text="Output: LSTM_IF subfolder", 
                 foreground='gray').grid(row=1, column=1, sticky=tk.W, pady=2)
        
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="▶ Start Training", 
                                    command=self.start_training, style='Accent.TButton')
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop Training", 
                                   command=self.stop_training, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="📊 Plot History", 
                  command=lambda: self.viz_type_var.set('training_history') or self.create_visualization()).pack(side=tk.LEFT, padx=5)
        
        progress_frame = ttk.LabelFrame(frame, text="Training Progress", padding="10")
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to train")
        self.progress_label.pack()
        
        # 로그 영역을 2개로 분할
        log_container = ttk.Frame(frame)
        log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 일반 Training Log (상단 50%)
        general_log_frame = ttk.LabelFrame(log_container, text="Training Log", padding="5")
        general_log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.train_log = scrolledtext.ScrolledText(general_log_frame, height=10, wrap=tk.WORD)
        self.train_log.pack(fill=tk.BOTH, expand=True)
        
        # Epoch Progress Log (하단 50%)
        epoch_log_frame = ttk.LabelFrame(log_container, text="Epoch Progress Log", padding="5")
        epoch_log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.epoch_log = scrolledtext.ScrolledText(epoch_log_frame, height=10, wrap=tk.WORD, 
                                                    font=('Courier New', 9))
        self.epoch_log.pack(fill=tk.BOTH, expand=True)
        
        # Epoch Log 스타일 태그 정의
        self.epoch_log.tag_config('epoch_start', foreground='#1976D2', font=('Courier New', 9, 'bold'))
        self.epoch_log.tag_config('epoch_end', foreground='#2E7D32', font=('Courier New', 9, 'bold'))
        self.epoch_log.tag_config('loss_improved', foreground='#2E7D32')
        self.epoch_log.tag_config('loss_worse', foreground='#C62828')
        self.epoch_log.tag_config('lr_change', foreground='#F57C00', font=('Courier New', 9, 'italic'))
    
    def create_detection_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔍 Detection")
        
        model_frame = ttk.LabelFrame(frame, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.selected_model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.selected_model_var, 
                                         width=40, state='readonly')
        self.model_combo.grid(row=0, column=1, padx=5, sticky=tk.W)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)
        
        ttk.Button(model_frame, text="🔄 Refresh Models", 
                  command=self.refresh_model_list).grid(row=0, column=2, padx=5)
        
        ttk.Label(model_frame, text="Auto-select: Latest model", 
                 foreground='gray').grid(row=1, column=1, sticky=tk.W, padx=5)
        
        threshold_frame = ttk.LabelFrame(frame, text="Threshold Settings", padding="10")
        threshold_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(threshold_frame, text="Model Default Threshold:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.model_threshold_label = ttk.Label(threshold_frame, text="No model loaded", 
                                               foreground='gray', font=('Arial', 10, 'bold'))
        self.model_threshold_label.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(threshold_frame, text="Custom Threshold:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.custom_threshold_var = tk.StringVar()
        self.custom_threshold_entry = ttk.Entry(threshold_frame, textvariable=self.custom_threshold_var, width=15)
        self.custom_threshold_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.use_custom_threshold_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(threshold_frame, text="Use Custom", 
                       variable=self.use_custom_threshold_var,
                       command=self.toggle_custom_threshold).grid(row=1, column=2, padx=5)
        
        ttk.Label(threshold_frame, text="Active Threshold:").grid(row=2, column=0, padx=5, sticky=tk.W)
        self.active_threshold_label = ttk.Label(threshold_frame, text="N/A", 
                                                foreground='blue', font=('Arial', 10, 'bold'))
        self.active_threshold_label.grid(row=2, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(threshold_frame, 
                 text="💡 Lower threshold = More sensitive (more anomalies detected)\n"
                      "   Higher threshold = Less sensitive (fewer anomalies detected)",
                 foreground='gray', font=('Arial', 8)).grid(row=3, column=0, columnspan=3, padx=5, pady=5)
        
        test_frame = ttk.LabelFrame(frame, text="Test Data", padding="10")
        test_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(test_frame, text="Test Folder:").grid(row=0, column=0, sticky=tk.W)
        self.test_path_var = tk.StringVar()
        ttk.Entry(test_frame, textvariable=self.test_path_var, width=70).grid(row=0, column=1, padx=5)
        ttk.Button(test_frame, text="Browse", command=self.select_test_folder).grid(row=0, column=2)
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="▶ Run Detection", command=self.run_detection).pack(side=tk.LEFT, padx=5)
        
        # ✅ 안내 문구 추가 (기존 버튼 2개 제거)
        info_label = ttk.Label(button_frame, 
                              text="📂 이상탐지 결과는 CSV파일과 HTML파일로 학습폴더 > LSTM_IF > reports 폴더에 자동 저장됩니다.",
                              foreground='#1976D2', font=('Arial', 9, 'bold'))
        info_label.pack(side=tk.LEFT, padx=15)
        
        result_frame = ttk.LabelFrame(frame, text="Detection Results", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=20, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
    
    def on_model_selected(self, event=None):
        selected_model_name = self.selected_model_var.get()
        if not selected_model_name or not self.training_data_dir:
            return
        
        try:
            models_dir = self.training_data_dir / "LSTM_IF"
            selected_model_path = models_dir / selected_model_name
            
            if selected_model_path.exists():
                # ✅ 수정: 메인 모델에 직접 로드
                if self.model is None:
                    self.model = EnhancedLSTMIFModel()
                
                self.model.load(selected_model_path)
                
                # ✅ 추가: threshold 유효성 검사
                if self.model.threshold is None or self.model.threshold <= 0:
                    raise ValueError(f"Invalid threshold: {self.model.threshold}")
                
                # Update UI
                self.model_threshold_label.config(
                    text=f"{self.model.threshold:.6f}",
                    foreground='green'
                )
                
                if not self.use_custom_threshold_var.get():
                    self.active_threshold_label.config(
                        text=f"{self.model.threshold:.6f} (Model Default)",
                        foreground='green'
                    )
                
                # ✅ 수정: 상세 로그
                self.log(f"✅ Model loaded: {selected_model_name}")
                self.log(f"   Threshold: {self.model.threshold:.6f}")
                self.log(f"   Range adjustments: {list(self.model.vacuum_range_thresholds.keys())}")
                
        except Exception as e:
            self.log(f"❌ Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            
            # ✅ 추가: UI 리셋
            self.model_threshold_label.config(text="Error", foreground='red')
            self.active_threshold_label.config(text="N/A", foreground='gray')
    
    def toggle_custom_threshold(self):
        if self.use_custom_threshold_var.get():
            self.custom_threshold_entry.config(state='normal')
            
            try:
                custom_value = float(self.custom_threshold_var.get())
                if custom_value <= 0:
                    messagebox.showerror("Error", "Threshold must be greater than 0")
                    self.use_custom_threshold_var.set(False)
                    return
                
                self.active_threshold_label.config(
                    text=f"{custom_value:.6f} (Custom)",
                    foreground='red'
                )
                self.log(f"Custom threshold activated: {custom_value:.6f}")
                
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")
                self.use_custom_threshold_var.set(False)
                return
        else:
            self.custom_threshold_entry.config(state='normal')
            model_threshold = self.model_threshold_label.cget("text")
            if model_threshold != "No model loaded":
                self.active_threshold_label.config(
                    text=f"{model_threshold} (Model Default)",
                    foreground='green'
                )
                self.log(f"Reverted to model default threshold: {model_threshold}")
    
    def get_active_threshold(self):
        if self.use_custom_threshold_var.get():
            try:
                return float(self.custom_threshold_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid custom threshold value")
                return None
        else:
            if self.model and self.model.threshold:
                return self.model.threshold
            return None        
    
    def refresh_model_list(self):
        if not self.training_data_dir:
            messagebox.showinfo("Info", "Please select training data folder first.")
            return
        
        models_dir = self.training_data_dir / "LSTM_IF"
        if not models_dir.exists():
            messagebox.showwarning("Warning", "No LSTM_IF folder found.")
            return
        
        model_files = sorted(list(models_dir.glob("lstmae_only_*.h5")) + list(models_dir.glob("lstmif_ensemble_*.h5")),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not model_files:
            messagebox.showwarning("Warning", "No trained models found.")
            return
        
        model_names = [f.name for f in model_files]
        self.model_combo['values'] = model_names
        
        if model_names:
            self.selected_model_var.set(model_names[0])
            self.on_model_selected()
            self.log(f"Models available: {len(model_names)}")
    
    def create_analysis_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Analysis")
        
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="Plot Type:").pack(side=tk.LEFT, padx=5)
        self.viz_type_var = tk.StringVar(value="training_history")
        viz_options = [
            'training_history',
            'error_distribution',
            'vacuum_range_distribution',
            'sensor_contribution',
            'threshold_analysis',
            'ensemble_comparison'
        ]
        ttk.Combobox(control_frame, textvariable=self.viz_type_var,
                    values=viz_options, width=25, state='readonly').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="📊 Generate", 
                  command=self.create_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 Save Plot", 
                  command=self.save_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📈 Export Stats", 
                  command=self.export_statistics).pack(side=tk.LEFT, padx=5)
        
        canvas_frame = ttk.LabelFrame(frame, text="Visualization", padding="10")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.viz_figure = plt.Figure(figsize=(14, 8), dpi=100)
        self.viz_canvas = FigureCanvasTkAgg(self.viz_figure, canvas_frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_model_manager_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📦 Models")
        
        # ========== 상단 통계 영역 (NEW) ==========
        stats_frame = ttk.LabelFrame(frame, text="📊 Quick Statistics", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_labels = {}
        stats_info = [
            ('total', 'Total: 0', 'Arial', 10, 'bold', 'black'),
            ('excellent', '🟢 Excellent: 0', 'Arial', 9, 'bold', '#1B5E20'),
            ('good', '🟢 Good: 0', 'Arial', 9, 'normal', '#2E7D32'),
            ('warning', '🟡 Warning: 0', 'Arial', 9, 'normal', '#F57C00'),
            ('poor', '🔴 Poor: 0', 'Arial', 9, 'bold', '#C62828'),
            ('best', 'Best Val Loss: N/A', 'Arial', 9, 'normal', '#1976D2')
        ]
        
        for i, (key, text, font_family, font_size, font_weight, fg) in enumerate(stats_info):
            label = ttk.Label(stats_frame, text=text, font=(font_family, font_size, font_weight), foreground=fg)
            label.grid(row=0, column=i, padx=12, pady=5)
            self.stats_labels[key] = label
        
        # ========== 컨트롤 버튼 ==========
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="🔄 Refresh", 
                  command=self.load_existing_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🗑 Delete Selected", 
                  command=self.delete_selected_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📋 View Details", 
                  command=self.show_model_details).pack(side=tk.LEFT, padx=5)
        
        # ✅ Export to CSV 버튼 제거, 안내 문구 추가
        info_label = ttk.Label(control_frame, 
                              text="📂 모델 정보는 CSV파일로 학습폴더 > LSTM_IF > reports 폴더에 자동 저장됩니다.",
                              foreground='#1976D2', font=('Arial', 9, 'bold'))
        info_label.pack(side=tk.LEFT, padx=15)
        
        # ========== Sub-Tabs (기존 코드 동일) ==========
        sub_notebook = ttk.Notebook(frame)
        sub_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ========== Tab 1: Overview ==========
        overview_frame = ttk.Frame(sub_notebook)
        sub_notebook.add(overview_frame, text="📋 Overview")
        
        columns_overview = (
            'Timestamp', 'Status', 'Val Loss', 'Val/Train', 'Converged',
            'Threshold', 'Best Epoch', 'Duration', 'Ensemble', 'Evaluation'
        )
        
        self.model_tree_overview = ttk.Treeview(overview_frame, columns=columns_overview, 
                                                show='headings', height=18)
        
        col_widths_overview = {
            'Timestamp': 95, 'Status': 95, 'Val Loss': 70, 'Val/Train': 75, 'Converged': 75,
            'Threshold': 75, 'Best Epoch': 75, 'Duration': 65, 'Ensemble': 70, 'Evaluation': 90
        }
        
        for col in columns_overview:
            self.model_tree_overview.heading(col, text=col, 
                command=lambda c=col: self.sort_models_by_tab(c, 'overview'))
            self.model_tree_overview.column(col, width=col_widths_overview[col], anchor='center')
        
        self.model_tree_overview.tag_configure('excellent', background='#E8F5E9', foreground='#1B5E20')
        self.model_tree_overview.tag_configure('good', background='#E3F2FD', foreground='#0D47A1')
        self.model_tree_overview.tag_configure('warning', background='#FFF3E0', foreground='#E65100')
        self.model_tree_overview.tag_configure('poor', background='#FFEBEE', foreground='#B71C1C')
        
        y_scroll_ov = ttk.Scrollbar(overview_frame, orient=tk.VERTICAL, command=self.model_tree_overview.yview)
        x_scroll_ov = ttk.Scrollbar(overview_frame, orient=tk.HORIZONTAL, command=self.model_tree_overview.xview)
        self.model_tree_overview.configure(yscroll=y_scroll_ov.set, xscroll=x_scroll_ov.set)
        
        self.model_tree_overview.grid(row=0, column=0, sticky='nsew')
        y_scroll_ov.grid(row=0, column=1, sticky='ns')
        x_scroll_ov.grid(row=1, column=0, sticky='ew')
        
        overview_frame.grid_rowconfigure(0, weight=1)
        overview_frame.grid_columnconfigure(0, weight=1)
        
        self.model_tree_overview.bind('<Double-1>', lambda e: self.show_model_details())
        
        # ========== Tab 2: Metrics ==========
        metrics_frame = ttk.Frame(sub_notebook)
        sub_notebook.add(metrics_frame, text="📊 Metrics")
        
        columns_metrics = (
            'Timestamp', 'Train Loss', 'Val Loss', 'Val/Train', 'Train MAE', 'Val MAE',
            'Train MSE', 'Val MSE', 'Val Std', 'Train Q95', 'Val Q95', 'Threshold'
        )
        
        self.model_tree_metrics = ttk.Treeview(metrics_frame, columns=columns_metrics, 
                                               show='headings', height=18)
        
        col_widths_metrics = {
            'Timestamp': 95, 'Train Loss': 75, 'Val Loss': 70, 'Val/Train': 75,
            'Train MAE': 75, 'Val MAE': 70, 'Train MSE': 75, 'Val MSE': 70,
            'Val Std': 70, 'Train Q95': 70, 'Val Q95': 70, 'Threshold': 75
        }
        
        for col in columns_metrics:
            self.model_tree_metrics.heading(col, text=col, 
                command=lambda c=col: self.sort_models_by_tab(c, 'metrics'))
            self.model_tree_metrics.column(col, width=col_widths_metrics[col], anchor='center')
        
        y_scroll_mt = ttk.Scrollbar(metrics_frame, orient=tk.VERTICAL, command=self.model_tree_metrics.yview)
        x_scroll_mt = ttk.Scrollbar(metrics_frame, orient=tk.HORIZONTAL, command=self.model_tree_metrics.xview)
        self.model_tree_metrics.configure(yscroll=y_scroll_mt.set, xscroll=x_scroll_mt.set)
        
        self.model_tree_metrics.grid(row=0, column=0, sticky='nsew')
        y_scroll_mt.grid(row=0, column=1, sticky='ns')
        x_scroll_mt.grid(row=1, column=0, sticky='ew')
        
        metrics_frame.grid_rowconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(0, weight=1)
        
        self.model_tree_metrics.bind('<Double-1>', lambda e: self.show_model_details())
        
        # ========== Tab 3: Architecture ==========
        arch_frame = ttk.Frame(sub_notebook)
        sub_notebook.add(arch_frame, text="🏗️ Architecture")
        
        columns_arch = (
            'Timestamp', 'Encoder L1', 'Encoder L2', 'Latent Dim', 'Decoder L1', 'Decoder L2',
            'Dropout', 'Rec Dropout', 'Learning Rate', 'Optimizer', 'Latent Act', 
            'L2 Reg', 'Clip Norm', 'BiDir', 'Batch Norm'
        )
        
        self.model_tree_arch = ttk.Treeview(arch_frame, columns=columns_arch, 
                                            show='headings', height=18)
        
        col_widths_arch = {
            'Timestamp': 95, 'Encoder L1': 75, 'Encoder L2': 75, 'Latent Dim': 80,
            'Decoder L1': 75, 'Decoder L2': 75, 'Dropout': 65, 'Rec Dropout': 85,
            'Learning Rate': 95, 'Optimizer': 80, 'Latent Act': 80,
            'L2 Reg': 70, 'Clip Norm': 75, 'BiDir': 50, 'Batch Norm': 85
        }
        
        for col in columns_arch:
            self.model_tree_arch.heading(col, text=col, 
                command=lambda c=col: self.sort_models_by_tab(c, 'arch'))
            self.model_tree_arch.column(col, width=col_widths_arch[col], anchor='center')
        
        y_scroll_ar = ttk.Scrollbar(arch_frame, orient=tk.VERTICAL, command=self.model_tree_arch.yview)
        x_scroll_ar = ttk.Scrollbar(arch_frame, orient=tk.HORIZONTAL, command=self.model_tree_arch.xview)
        self.model_tree_arch.configure(yscroll=y_scroll_ar.set, xscroll=x_scroll_ar.set)
        
        self.model_tree_arch.grid(row=0, column=0, sticky='nsew')
        y_scroll_ar.grid(row=0, column=1, sticky='ns')
        x_scroll_ar.grid(row=1, column=0, sticky='ew')
        
        arch_frame.grid_rowconfigure(0, weight=1)
        arch_frame.grid_columnconfigure(0, weight=1)
        
        self.model_tree_arch.bind('<Double-1>', lambda e: self.show_model_details())
        
        # ========== Tab 4: Training Config ==========
        config_frame = ttk.Frame(sub_notebook)
        sub_notebook.add(config_frame, text="⚙️ Training Config")
        
        columns_config = (
            'Timestamp', 'Batch Size', 'Total Epochs', 'Trained Epochs', 'Augment',
            'Patience', 'Min Delta', 'Val Split', 'LR Schedule', 'Save Ckpt'
        )
        
        self.model_tree_config = ttk.Treeview(config_frame, columns=columns_config, 
                                              show='headings', height=18)
        
        col_widths_config = {
            'Timestamp': 95, 'Batch Size': 75, 'Total Epochs': 85, 'Trained Epochs': 95,
            'Augment': 65, 'Patience': 65, 'Min Delta': 75, 'Val Split': 70,
            'LR Schedule': 85, 'Save Ckpt': 70
        }
        
        for col in columns_config:
            self.model_tree_config.heading(col, text=col, 
                command=lambda c=col: self.sort_models_by_tab(c, 'config'))
            self.model_tree_config.column(col, width=col_widths_config[col], anchor='center')
        
        y_scroll_cf = ttk.Scrollbar(config_frame, orient=tk.VERTICAL, command=self.model_tree_config.yview)
        x_scroll_cf = ttk.Scrollbar(config_frame, orient=tk.HORIZONTAL, command=self.model_tree_config.xview)
        self.model_tree_config.configure(yscroll=y_scroll_cf.set, xscroll=x_scroll_cf.set)
        
        self.model_tree_config.grid(row=0, column=0, sticky='nsew')
        y_scroll_cf.grid(row=0, column=1, sticky='ns')
        x_scroll_cf.grid(row=1, column=0, sticky='ew')
        
        config_frame.grid_rowconfigure(0, weight=1)
        config_frame.grid_columnconfigure(0, weight=1)
        
        self.model_tree_config.bind('<Double-1>', lambda e: self.show_model_details())
        
        # ========== 기존 self.model_tree는 호환성 유지 ==========
        self.model_tree = self.model_tree_overview
        
        # 초기 로드
        self.load_existing_models()
        
    def auto_export_models_csv(self):
        """
        훈련 완료 후 자동으로 모델 정보를 CSV로 저장
        - 저장 위치: {training_data_dir}/LSTM_IF/reports/
        - 파일명: lstmif_models_full_YYMMDD_HHMM.csv
        - 기존 lstmif_models_full_*.csv 파일 삭제 후 새 파일 저장 (폴더에 1개만 유지)
        """
        if not self.loaded_models_info:
            self.log("⚠️ No models to export")
            return False
        
        if not self.training_data_dir:
            self.log("❌ Training data folder not set")
            return False
        
        try:
            # reports 폴더 생성
            report_dir = self.training_data_dir / "LSTM_IF" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # ✅ 기존 lstmif_models_full_*.csv 파일 모두 삭제
            existing_files = list(report_dir.glob("lstmif_models_full_*.csv"))
            for old_file in existing_files:
                try:
                    old_file.unlink()
                    self.log(f"🗑️ Deleted old file: {old_file.name}")
                except Exception as e:
                    self.log(f"⚠️ Could not delete {old_file.name}: {str(e)}")
            
            # 새 파일명 생성
            timestamp = datetime.now().strftime('%y%m%d_%H%M')
            file_path = report_dir / f"lstmif_models_full_{timestamp}.csv"
            
            self.log(f"📄 Saving models CSV: {file_path.name}...")
            
            # CSV 데이터 생성 (기존 export_models_to_csv()와 동일한 로직)
            export_data = []
            
            for timestamp_key, model_info in sorted(self.loaded_models_info.items(), 
                                               key=lambda x: x[0], reverse=True):
                metadata = model_info['metadata']
                
                train_stats = metadata.get('train_stats', {})
                val_stats = metadata.get('val_stats', {})
                params = metadata.get('params', {})
                training_params = metadata.get('training_params', {})
                history = metadata.get('history', {})
                
                train_mse = train_stats.get('mean', 0)
                val_mse = val_stats.get('mean', 0)
                
                train_loss_final = history['loss'][-1] if history.get('loss') else train_mse
                val_loss_final = history['val_loss'][-1] if history.get('val_loss') else val_mse
                
                val_train_ratio = val_loss_final / train_loss_final if train_loss_final > 0 else 0
                
                if history.get('val_loss') and len(history['val_loss']) >= 20:
                    val_loss_std = np.std(history['val_loss'][-20:])
                else:
                    val_loss_std = 0
                
                best_epoch = metadata.get('best_epoch', 0)
                epochs_trained = training_params.get('epochs_trained', 0)
                early_gap = epochs_trained - best_epoch
                converged = early_gap <= training_params.get('patience', 20) + 5
                
                train_mae = history['mae'][-1] if history.get('mae') else train_mse ** 0.5
                val_mae = history['val_mae'][-1] if history.get('val_mae') else val_mse ** 0.5
                
                train_q95 = train_stats.get('q95', train_mse * 1.5)
                val_q95 = val_stats.get('q95', val_mse * 1.5)
                
                if val_loss_final < 0.005 and val_train_ratio < 1.8 and converged:
                    status = "Excellent"
                elif val_loss_final < 0.008 and val_train_ratio < 2.0 and converged:
                    status = "Good"
                elif val_loss_final < 0.015 and val_train_ratio < 2.5:
                    status = "Warning"
                else:
                    status = "Poor"
                
                train_time = metadata.get('training_time', 0)
                if train_time >= 3600:
                    duration_str = f"{train_time/3600:.1f}h"
                elif train_time >= 60:
                    duration_str = f"{train_time/60:.1f}m"
                else:
                    duration_str = f"{train_time:.0f}s"
                
                model_id = metadata.get('model_id', 'unknown')
                ensemble_path = self.training_data_dir / "LSTM_IF" / f"ensemble_{model_id}.pkl"
                ensemble = "Yes" if ensemble_path.exists() else "No"
                
                encoder_units = params.get('encoder_units', [64, 32])
                latent_dim = params.get('latent_dim', 16)
                dropout = params.get('dropout', 0.25)
                recurrent_dropout = params.get('recurrent_dropout', 0.15)
                lr = params.get('lr', 0.001)
                optimizer = params.get('optimizer', 'adam')
                latent_activation = params.get('latent_activation', 'tanh')
                l2_reg = params.get('l2_reg', 0.001)
                clip_norm = params.get('clip_norm', 1.0)
                use_bidir = params.get('use_bidirectional', True)
                use_batch_norm = params.get('use_batch_norm', True)
                
                batch_size = training_params.get('batch_size', 'N/A')
                total_epochs = training_params.get('epochs', 'N/A')
                augmentation = training_params.get('augmentation', 'N/A')
                patience = training_params.get('patience', 'N/A')
                min_delta = training_params.get('min_delta', 'N/A')
                val_split = training_params.get('validation_split', 'N/A')
                lr_schedule = training_params.get('lr_schedule', 'N/A')
                save_checkpoints = training_params.get('save_checkpoints', 'N/A')
                
                row = {
                    'Timestamp': timestamp_key,
                    'Status': status,
                    'Val_Loss': f"{val_loss_final:.6f}",
                    'Val_Train_Ratio': f"{val_train_ratio:.2f}",
                    'Converged': 'Yes' if converged else 'No',
                    'Threshold': f"{metadata.get('threshold', 0):.6f}",
                    'Best_Epoch': f"{best_epoch}/{epochs_trained}",
                    'Duration': duration_str,
                    'Ensemble': ensemble,
                    
                    'Train_Loss': f"{train_loss_final:.6f}",
                    'Train_MAE': f"{train_mae:.6f}",
                    'Val_MAE': f"{val_mae:.6f}",
                    'Train_MSE': f"{train_mse:.6f}",
                    'Val_MSE': f"{val_mse:.6f}",
                    'Val_Std': f"{val_loss_std:.6f}",
                    'Train_Q95': f"{train_q95:.6f}",
                    'Val_Q95': f"{val_q95:.6f}",
                    
                    'Encoder_L1': encoder_units[0] if len(encoder_units) > 0 else 'N/A',
                    'Encoder_L2': encoder_units[1] if len(encoder_units) > 1 else 'N/A',
                    'Latent_Dim': latent_dim,
                    'Decoder_L1': encoder_units[1] if len(encoder_units) > 1 else 'N/A',
                    'Decoder_L2': encoder_units[0] if len(encoder_units) > 0 else 'N/A',
                    'Dropout': f"{dropout:.2f}",
                    'Recurrent_Dropout': f"{recurrent_dropout:.2f}",
                    'Learning_Rate': f"{lr:.6f}",
                    'Optimizer': optimizer,
                    'Latent_Activation': latent_activation,
                    'L2_Regularization': f"{l2_reg:.4f}",
                    'Gradient_Clip_Norm': f"{clip_norm:.2f}",
                    'Bidirectional': 'Yes' if use_bidir else 'No',
                    'Batch_Normalization': 'Yes' if use_batch_norm else 'No',
                    
                    'Batch_Size': batch_size,
                    'Total_Epochs': total_epochs,
                    'Trained_Epochs': epochs_trained,
                    'Augmentation': augmentation,
                    'Patience': patience,
                    'Min_Delta': f"{min_delta:.6f}" if isinstance(min_delta, (int, float)) else min_delta,
                    'Val_Split': f"{val_split:.2f}" if isinstance(val_split, (int, float)) else val_split,
                    'LR_Schedule': lr_schedule,
                    'Save_Checkpoint': 'Yes' if save_checkpoints is True else ('No' if save_checkpoints is False else 'N/A')
                }
                
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            self.log(f"✅ Models CSV saved: {file_path.name} ({len(export_data)} models)")
            return True
            
        except Exception as e:
            self.log(f"❌ Models CSV save failed: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return False        
  
    def create_evaluation_tab(self):
        """Create comprehensive model evaluation tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Evaluation")
        
        # Control Frame
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="Select Model:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.eval_model_var = tk.StringVar()
        self.eval_model_combo = ttk.Combobox(control_frame, textvariable=self.eval_model_var, 
                                              width=40, state='readonly')
        self.eval_model_combo.pack(side=tk.LEFT, padx=5)
        self.eval_model_combo.bind('<<ComboboxSelected>>', self.on_eval_model_selected)
        
        ttk.Button(control_frame, text="🔄 Refresh", 
                   command=self.refresh_eval_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📥 Load Evaluation", 
                   command=self.load_evaluation_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 Save Evaluation", 
                   command=self.save_model_evaluation).pack(side=tk.LEFT, padx=5)
        
        # Paned Window for split view
        paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left Panel - Input Form
        left_panel = ttk.LabelFrame(paned, text="Model Evaluation Input", padding="10")
        paned.add(left_panel, weight=1)
        
        # Model Info Display
        info_frame = ttk.LabelFrame(left_panel, text="Selected Model Info", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.eval_model_info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, 
                                             font=('Courier New', 9), state='disabled')
        self.eval_model_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Input Form
        input_frame = ttk.LabelFrame(left_panel, text="Detection Results Input", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Anomaly Data Results
        anomaly_frame = ttk.LabelFrame(input_frame, text="🔴 Anomaly Data Detection", padding="10")
        anomaly_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(anomaly_frame, text="Total Anomaly Samples:", 
                  font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.total_anomaly_var = tk.IntVar(value=0)
        ttk.Spinbox(anomaly_frame, from_=0, to=10000, textvariable=self.total_anomaly_var, 
                    width=15).grid(row=0, column=1, padx=5, pady=3)
        
        ttk.Label(anomaly_frame, text="Detected as Anomaly:", 
                  foreground='#C62828').grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.anomaly_detected_anomaly_var = tk.IntVar(value=0)
        ttk.Spinbox(anomaly_frame, from_=0, to=10000, textvariable=self.anomaly_detected_anomaly_var, 
                    width=15).grid(row=1, column=1, padx=5, pady=3)
        
        ttk.Label(anomaly_frame, text="Detected as Normal:", 
                  foreground='#2E7D32').grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.anomaly_detected_normal_var = tk.IntVar(value=0)
        ttk.Spinbox(anomaly_frame, from_=0, to=10000, textvariable=self.anomaly_detected_normal_var, 
                    width=15).grid(row=2, column=1, padx=5, pady=3)
        
        # Auto-calculate button
        ttk.Button(anomaly_frame, text="↻ Auto Calculate", 
                   command=self.auto_calculate_anomaly).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Normal Data Results
        normal_frame = ttk.LabelFrame(input_frame, text="🟢 Normal Data Detection", padding="10")
        normal_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(normal_frame, text="Total Normal Samples:", 
                  font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.total_normal_var = tk.IntVar(value=0)
        ttk.Spinbox(normal_frame, from_=0, to=10000, textvariable=self.total_normal_var, 
                    width=15).grid(row=0, column=1, padx=5, pady=3)
        
        ttk.Label(normal_frame, text="Detected as Normal:", 
                  foreground='#2E7D32').grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.normal_detected_normal_var = tk.IntVar(value=0)
        ttk.Spinbox(normal_frame, from_=0, to=10000, textvariable=self.normal_detected_normal_var, 
                    width=15).grid(row=1, column=1, padx=5, pady=3)
        
        ttk.Label(normal_frame, text="Detected as Anomaly:", 
                  foreground='#C62828').grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.normal_detected_anomaly_var = tk.IntVar(value=0)
        ttk.Spinbox(normal_frame, from_=0, to=10000, textvariable=self.normal_detected_anomaly_var, 
                    width=15).grid(row=2, column=1, padx=5, pady=3)
        
        ttk.Button(normal_frame, text="↻ Auto Calculate", 
                   command=self.auto_calculate_normal).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Metrics Preview
        metrics_frame = ttk.LabelFrame(input_frame, text="📈 Calculated Metrics Preview", padding="10")
        metrics_frame.pack(fill=tk.X, pady=5)
        
        self.metrics_preview_text = tk.Text(metrics_frame, height=8, wrap=tk.WORD,
                                             font=('Courier New', 9), state='disabled',
                                             background='#f0f0f0')
        self.metrics_preview_text.pack(fill=tk.BOTH, expand=True)
        
        # Bind variables to update preview
        for var in [self.total_anomaly_var, self.anomaly_detected_anomaly_var, self.anomaly_detected_normal_var,
                    self.total_normal_var, self.normal_detected_normal_var, self.normal_detected_anomaly_var]:
            var.trace('w', lambda *args: self.update_metrics_preview())
        
        # Action Buttons
        action_frame = ttk.Frame(left_panel)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="✅ Save Evaluation", 
                   command=self.save_model_evaluation,
                   style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🗑 Clear Input", 
                   command=self.clear_eval_input).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📋 Copy from Detection", 
                   command=self.copy_from_detection_results).pack(side=tk.LEFT, padx=5)
        
        # Right Panel - Results View
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=2)
        
        # Notebook for different views
        results_notebook = ttk.Notebook(right_panel)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Models Overview
        overview_frame = ttk.Frame(results_notebook)
        results_notebook.add(overview_frame, text="📋 Models Overview")
        
        # Control buttons for overview
        overview_control = ttk.Frame(overview_frame)
        overview_control.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(overview_control, text="🔄 Refresh", 
                   command=self.refresh_evaluation_overview).pack(side=tk.LEFT, padx=5)
        ttk.Button(overview_control, text="📊 Compare All", 
                   command=self.show_comparison_chart).pack(side=tk.LEFT, padx=5)
        ttk.Button(overview_control, text="📄 Export Report", 
                   command=self.export_evaluation_report).pack(side=tk.LEFT, padx=5)
        
        # Treeview for overview
        tree_frame = ttk.Frame(overview_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Model', 'Status', 'Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'FNR', 'Total Samples')
        self.eval_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
        
        col_widths = {
            'Model': 150, 'Status': 80, 'Accuracy': 80, 'Precision': 80,
            'Recall': 80, 'F1': 80, 'FPR': 80, 'FNR': 80, 'Total Samples': 100
        }
        
        for col in columns:
            self.eval_tree.heading(col, text=col, command=lambda c=col: self.sort_evaluation_by(c))
            self.eval_tree.column(col, width=col_widths[col], anchor='center')
        
        eval_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.eval_tree.yview)
        eval_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.eval_tree.xview)
        self.eval_tree.configure(yscroll=eval_scroll_y.set, xscroll=eval_scroll_x.set)
        
        self.eval_tree.grid(row=0, column=0, sticky='nsew')
        eval_scroll_y.grid(row=0, column=1, sticky='ns')
        eval_scroll_x.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        self.eval_tree.bind('<Double-1>', lambda e: self.show_detailed_evaluation())
        
        # Tab 2: Comparison Charts
        comparison_frame = ttk.Frame(results_notebook)
        results_notebook.add(comparison_frame, text="📊 Performance Comparison")
        
        comparison_control = ttk.Frame(comparison_frame)
        comparison_control.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(comparison_control, text="Chart Type:").pack(side=tk.LEFT, padx=5)
        self.comparison_chart_var = tk.StringVar(value='accuracy_comparison')
        chart_types = ['accuracy_comparison', 'precision_recall', 'error_rates', 'confusion_matrix', 'roc_analysis']
        ttk.Combobox(comparison_control, textvariable=self.comparison_chart_var,
                    values=chart_types, width=20, state='readonly').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(comparison_control, text="📊 Generate Chart", 
                   command=self.generate_comparison_chart).pack(side=tk.LEFT, padx=5)
        ttk.Button(comparison_control, text="💾 Save Chart", 
                   command=self.save_comparison_chart).pack(side=tk.LEFT, padx=5)
        
        self.comparison_figure = plt.Figure(figsize=(12, 8), dpi=100)
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_figure, comparison_frame)
        self.comparison_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 3: Detailed Metrics
        details_frame = ttk.Frame(results_notebook)
        results_notebook.add(details_frame, text="📈 Detailed Metrics")
        
        self.detailed_metrics_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD,
                                                                font=('Courier New', 9))
        self.detailed_metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize
        self.refresh_eval_models()
        self.refresh_evaluation_overview()
    
    def refresh_eval_models(self):
        """Refresh model list in evaluation tab"""
        if not self.training_data_dir:
            return
        
        models_dir = self.training_data_dir / "LSTM_IF"
        if not models_dir.exists():
            return
        
        model_files = sorted(list(models_dir.glob("lstmae_only_*.h5")) + list(models_dir.glob("lstmif_ensemble_*.h5")),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        model_names = [f.name for f in model_files]
        self.eval_model_combo['values'] = model_names
        
        if model_names:
            self.eval_model_var.set(model_names[0])
            self.on_eval_model_selected()
    
    def on_eval_model_selected(self, event=None):
        """Handle evaluation model selection"""
        selected = self.eval_model_var.get()
        if not selected or not self.training_data_dir:
            return
        
        try:
            models_dir = self.training_data_dir / "LSTM_IF"
            model_path = models_dir / selected
            metadata_path = models_dir / f"{selected.replace('.h5', '_metadata.pkl')}"
            
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Display model info
                self.eval_model_info_text.config(state='normal')
                self.eval_model_info_text.delete(1.0, tk.END)
                
                info_lines = [
                    f"Model: {selected}",
                    f"Created: {metadata.get('created_at', 'N/A')}",
                    f"Threshold: {metadata.get('threshold', 0):.6f}",
                    f"Best Epoch: {metadata.get('best_epoch', 0)}",
                    f"Train MSE: {metadata.get('train_stats', {}).get('mean', 0):.6f}",
                    f"Val MSE: {metadata.get('val_stats', {}).get('mean', 0):.6f}"
                ]
                
                self.eval_model_info_text.insert(1.0, '\n'.join(info_lines))
                self.eval_model_info_text.config(state='disabled')
                
                # ✅ Extract timestamp from filename for dictionary key
                model_stem = selected.replace('.h5', '')
                timestamp_key = model_stem.replace('lstmae_only_', '').replace('lstmif_ensemble_', '')
                
                # Load existing evaluation if available (check both metadata and evaluation_data)
                # Priority 1: Check metadata (most reliable)
                eval_data = metadata.get('evaluation', None)
                
                # Priority 2: Check evaluation_data dictionary
                if not eval_data and timestamp_key in self.evaluation_data:
                    eval_data = self.evaluation_data[timestamp_key]
                
                if eval_data:
                    self.total_anomaly_var.set(eval_data.get('total_anomaly', 0))
                    self.anomaly_detected_anomaly_var.set(eval_data.get('true_positive', 0))
                    self.anomaly_detected_normal_var.set(eval_data.get('false_negative', 0))
                    self.total_normal_var.set(eval_data.get('total_normal', 0))
                    self.normal_detected_normal_var.set(eval_data.get('true_negative', 0))
                    self.normal_detected_anomaly_var.set(eval_data.get('false_positive', 0))
                else:
                    self.clear_eval_input()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model info:\n{str(e)}")
    
    def auto_calculate_anomaly(self):
        """Auto-calculate anomaly detection counts"""
        total = self.total_anomaly_var.get()
        detected_anomaly = self.anomaly_detected_anomaly_var.get()
        
        if total >= detected_anomaly:
            self.anomaly_detected_normal_var.set(total - detected_anomaly)
        else:
            messagebox.showwarning("Warning", "Detected anomaly count cannot exceed total anomaly samples!")
    
    def auto_calculate_normal(self):
        """Auto-calculate normal detection counts"""
        total = self.total_normal_var.get()
        detected_normal = self.normal_detected_normal_var.get()
        
        if total >= detected_normal:
            self.normal_detected_anomaly_var.set(total - detected_normal)
        else:
            messagebox.showwarning("Warning", "Detected normal count cannot exceed total normal samples!")
    
    def update_metrics_preview(self):
        """Update metrics preview in real-time"""
        try:
            tp = self.anomaly_detected_anomaly_var.get()
            fn = self.anomaly_detected_normal_var.get()
            tn = self.normal_detected_normal_var.get()
            fp = self.normal_detected_anomaly_var.get()
            
            total = tp + fn + tn + fp
            
            if total == 0:
                self.metrics_preview_text.config(state='normal')
                self.metrics_preview_text.delete(1.0, tk.END)
                self.metrics_preview_text.insert(1.0, "Enter detection results to see metrics preview...")
                self.metrics_preview_text.config(state='disabled')
                return
            
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            preview_text = f"""
    ╔══════════════════════════════════════════════╗
    ║         PERFORMANCE METRICS PREVIEW          ║
    ╚══════════════════════════════════════════════╝
    
    Confusion Matrix:
      TP (True Positive):  {tp:>6}   FN (False Negative): {fn:>6}
      FP (False Positive): {fp:>6}   TN (True Negative):  {tn:>6}
    
    Performance Metrics:
      • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)
      • Precision: {precision:.4f} ({precision*100:.2f}%)
      • Recall:    {recall:.4f} ({recall*100:.2f}%)
      • F1 Score:  {f1:.4f}
      • FPR:       {fpr:.4f} ({fpr*100:.2f}%)
      • FNR:       {fnr:.4f} ({fnr*100:.2f}%)
    
    Total Samples: {total}
            """
            
            self.metrics_preview_text.config(state='normal')
            self.metrics_preview_text.delete(1.0, tk.END)
            self.metrics_preview_text.insert(1.0, preview_text.strip())
            self.metrics_preview_text.config(state='disabled')
            
        except Exception as e:
            pass
    
    def save_model_evaluation(self):
        """Save evaluation data for selected model"""
        selected = self.eval_model_var.get()
        if not selected:
            messagebox.showwarning("Warning", "Please select a model first.")
            return
        
        try:
            tp = self.anomaly_detected_anomaly_var.get()
            fn = self.anomaly_detected_normal_var.get()
            tn = self.normal_detected_normal_var.get()
            fp = self.normal_detected_anomaly_var.get()
            
            total_anomaly = self.total_anomaly_var.get()
            total_normal = self.total_normal_var.get()
            
            if tp + fn != total_anomaly:
                messagebox.showerror("Error", "Anomaly data counts don't match!\n"
                                   f"TP + FN ({tp + fn}) should equal Total Anomaly ({total_anomaly})")
                return
            
            if tn + fp != total_normal:
                messagebox.showerror("Error", "Normal data counts don't match!\n"
                                   f"TN + FP ({tn + fp}) should equal Total Normal ({total_normal})")
                return
            
            total = tp + fn + tn + fp
            if total == 0:
                messagebox.showerror("Error", "No data entered!")
                return
            
            # Calculate metrics
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            eval_result = {
                'timestamp': datetime.now().isoformat(),
                'total_anomaly': total_anomaly,
                'total_normal': total_normal,
                'true_positive': tp,
                'false_negative': fn,
                'true_negative': tn,
                'false_positive': fp,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'fpr': fpr,
                'fnr': fnr,
                'specificity': specificity,
                'total_samples': total
            }
            
            # ✅ Extract timestamp from filename for dictionary key
            # selected = "lstmae_only_250120_1430.h5" or "lstmif_ensemble_250120_1430.h5"
            # We need key = "250120_1430"
            model_stem = selected.replace('.h5', '')
            timestamp_key = model_stem.replace('lstmae_only_', '').replace('lstmif_ensemble_', '')
            
            # Save to evaluation_data dictionary with timestamp key
            self.evaluation_data[timestamp_key] = eval_result
            
            # ✅ Save to model metadata file
            if self.training_data_dir:
                models_dir = self.training_data_dir / "LSTM_IF"
                model_path = models_dir / selected
                metadata_path = models_dir / f"{model_path.stem}_metadata.pkl"
                
                if metadata_path.exists():
                    # Load existing metadata
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    
                    # Add evaluation results to metadata
                    metadata['evaluation'] = eval_result
                    
                    # Save updated metadata
                    with open(metadata_path, 'wb') as f:
                        pickle.dump(metadata, f)
                    
                    self.log(f"✅ Evaluation saved to model metadata: {selected}")
                
                # Also save to evaluations.pkl for backward compatibility
                eval_file = models_dir / "evaluations.pkl"
                with open(eval_file, 'wb') as f:
                    pickle.dump(self.evaluation_data, f)
            
            # Refresh both evaluation overview and models overview
            self.refresh_evaluation_overview()
            self.load_existing_models()  # ✅ Models Overview 새로고침
            
            messagebox.showinfo("Success", f"Evaluation saved for model: {selected}\n\n"
                              f"Results:\n"
                              f"- Accuracy: {accuracy:.4f}\n"
                              f"- Precision: {precision:.4f}\n"
                              f"- Recall: {recall:.4f}\n"
                              f"- F1 Score: {f1:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save evaluation:\n{str(e)}")
    
    def clear_eval_input(self):
        """Clear evaluation input fields"""
        self.total_anomaly_var.set(0)
        self.anomaly_detected_anomaly_var.set(0)
        self.anomaly_detected_normal_var.set(0)
        self.total_normal_var.set(0)
        self.normal_detected_normal_var.set(0)
        self.normal_detected_anomaly_var.set(0)
    
    def copy_from_detection_results(self):
        """Copy results from detection tab"""
        if not self.current_results:
            messagebox.showwarning("Warning", "No detection results available in Detection tab.")
            return
        
        # Count results
        total_files = len(self.current_results)
        anomaly_count = sum(1 for r in self.current_results if r['is_anomaly'])
        normal_count = total_files - anomaly_count
        
        # Ask user which data type
        response = messagebox.askquestion(
            "Copy Detection Results",
            f"Detection results:\n"
            f"- Total: {total_files}\n"
            f"- Detected as Anomaly: {anomaly_count}\n"
            f"- Detected as Normal: {normal_count}\n\n"
            f"Are these results from Anomaly data?\n"
            f"(Click 'Yes' for Anomaly data, 'No' for Normal data)"
        )
        
        if response == 'yes':
            # These are anomaly data results
            self.total_anomaly_var.set(total_files)
            self.anomaly_detected_anomaly_var.set(anomaly_count)
            self.anomaly_detected_normal_var.set(normal_count)
        else:
            # These are normal data results
            self.total_normal_var.set(total_files)
            self.normal_detected_normal_var.set(normal_count)
            self.normal_detected_anomaly_var.set(anomaly_count)
        
        messagebox.showinfo("Success", "Detection results copied successfully!")
    
    def load_evaluation_data(self):
        """Load evaluation data from file"""
        if not self.training_data_dir:
            messagebox.showwarning("Warning", "Please select training data folder first.")
            return
        
        eval_file = self.training_data_dir / "LSTM_IF" / "evaluations.pkl"
        if eval_file.exists():
            try:
                with open(eval_file, 'rb') as f:
                    self.evaluation_data = pickle.load(f)
                self.refresh_evaluation_overview()
                messagebox.showinfo("Success", f"Loaded evaluations for {len(self.evaluation_data)} models.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load evaluations:\n{str(e)}")
        else:
            messagebox.showinfo("Info", "No saved evaluations found.")
    
    def save_evaluation_data(self):
        """Save all evaluation data to file"""
        if not self.evaluation_data:
            messagebox.showwarning("Warning", "No evaluation data to save.")
            return
        
        if not self.training_data_dir:
            messagebox.showwarning("Warning", "Please select training data folder first.")
            return
        
        try:
            eval_file = self.training_data_dir / "LSTM_IF" / "evaluations.pkl"
            with open(eval_file, 'wb') as f:
                pickle.dump(self.evaluation_data, f)
            messagebox.showinfo("Success", f"Saved evaluations for {len(self.evaluation_data)} models.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save evaluations:\n{str(e)}")
    
    def refresh_evaluation_overview(self):
        """Refresh evaluation overview table"""
        self.eval_tree.delete(*self.eval_tree.get_children())
        
        if not self.training_data_dir:
            return
        
        models_dir = self.training_data_dir / "LSTM_IF"
        if not models_dir.exists():
            return
        
        model_files = sorted(list(models_dir.glob("lstmae_only_*.h5")) + list(models_dir.glob("lstmif_ensemble_*.h5")),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for model_file in model_files:
            model_name = model_file.stem.replace('lstmae_only_', '').replace('lstmif_ensemble_', '')
            
            if model_name in self.evaluation_data:
                eval_data = self.evaluation_data[model_name]
                
                self.eval_tree.insert('', 'end', iid=model_name, values=(
                    model_name,
                    '✅ Evaluated',
                    f"{eval_data['accuracy']:.4f}",
                    f"{eval_data['precision']:.4f}",
                    f"{eval_data['recall']:.4f}",
                    f"{eval_data['f1_score']:.4f}",
                    f"{eval_data['fpr']:.4f}",
                    f"{eval_data['fnr']:.4f}",
                    eval_data['total_samples']
                ), tags=('evaluated',))
            else:
                self.eval_tree.insert('', 'end', iid=model_name, values=(
                    model_name,
                    '⚠️ Not Evaluated',
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                ), tags=('not_evaluated',))
        
        # Configure tags
        self.eval_tree.tag_configure('evaluated', background='#E8F5E9')
        self.eval_tree.tag_configure('not_evaluated', background='#FFF3E0')
    
    def sort_evaluation_by(self, column):
        """Sort evaluation table by column"""
        items = [(self.eval_tree.set(item, column), item) for item in self.eval_tree.get_children('')]
        
        try:
            items.sort(key=lambda x: float(x[0].replace('N/A', '-1').replace('✅ Evaluated', '1').replace('⚠️ Not Evaluated', '0')))
        except:
            items.sort()
        
        for index, (val, item) in enumerate(items):
            self.eval_tree.move(item, '', index)
    
    def show_detailed_evaluation(self):
        """Show detailed evaluation for selected model"""
        selection = self.eval_tree.selection()
        if not selection:
            return
        
        model_name = selection[0]
        if model_name not in self.evaluation_data:
            messagebox.showinfo("Info", f"No evaluation data for model: {model_name}")
            return
        
        eval_data = self.evaluation_data[model_name]
        
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Detailed Evaluation - {model_name}")
        detail_window.geometry("700x600")
        
        detail_text = scrolledtext.ScrolledText(detail_window, wrap=tk.WORD, font=('Courier New', 10))
        detail_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        detail_content = f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                     DETAILED EVALUATION REPORT                       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Model: {model_name}
    Evaluation Date: {eval_data['timestamp']}
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DATA DISTRIBUTION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Total Samples:        {eval_data['total_samples']}
      • Anomaly Samples:  {eval_data['total_anomaly']} ({eval_data['total_anomaly']/eval_data['total_samples']*100:.1f}%)
      • Normal Samples:   {eval_data['total_normal']} ({eval_data['total_normal']/eval_data['total_samples']*100:.1f}%)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CONFUSION MATRIX
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
                        Predicted
                    Anomaly    Normal
        Actual  ┌──────────┬──────────┐
        Anomaly │   {eval_data['true_positive']:>4}   │   {eval_data['false_negative']:>4}   │
                ├──────────┼──────────┤
        Normal  │   {eval_data['false_positive']:>4}   │   {eval_data['true_negative']:>4}   │
                └──────────┴──────────┘
    
      ✓ True Positive (TP):   {eval_data['true_positive']:>6}  (Correctly detected anomalies)
      ✗ False Negative (FN):  {eval_data['false_negative']:>6}  (Missed anomalies)
      ✓ True Negative (TN):   {eval_data['true_negative']:>6}  (Correctly detected normal)
      ✗ False Positive (FP):  {eval_data['false_positive']:>6}  (False alarms)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PERFORMANCE METRICS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Overall Accuracy:     {eval_data['accuracy']:.4f} ({eval_data['accuracy']*100:.2f}%)
      → Percentage of correct predictions
    
    Precision:            {eval_data['precision']:.4f} ({eval_data['precision']*100:.2f}%)
      → TP / (TP + FP) - Accuracy of anomaly predictions
      → How many predicted anomalies were actually anomalies
    
    Recall (Sensitivity): {eval_data['recall']:.4f} ({eval_data['recall']*100:.2f}%)
      → TP / (TP + FN) - Detection rate
      → How many actual anomalies were detected
    
    F1 Score:             {eval_data['f1_score']:.4f}
      → Harmonic mean of Precision and Recall
      → Balanced measure of model performance
    
    Specificity:          {eval_data['specificity']:.4f} ({eval_data['specificity']*100:.2f}%)
      → TN / (TN + FP) - True negative rate
      → How many normal samples were correctly identified
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ERROR RATES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    False Positive Rate:  {eval_data['fpr']:.4f} ({eval_data['fpr']*100:.2f}%)
      → FP / (FP + TN) - False alarm rate
      → Percentage of normal samples incorrectly flagged as anomalies
    
    False Negative Rate:  {eval_data['fnr']:.4f} ({eval_data['fnr']*100:.2f}%)
      → FN / (FN + TP) - Miss rate
      → Percentage of anomalies that were missed
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    INTERPRETATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    """
        
        # Add interpretation
        if eval_data['accuracy'] >= 0.95:
            detail_content += "✓ Excellent overall performance\n"
        elif eval_data['accuracy'] >= 0.90:
            detail_content += "✓ Very good overall performance\n"
        elif eval_data['accuracy'] >= 0.85:
            detail_content += "◐ Good overall performance\n"
        else:
            detail_content += "✗ Performance needs improvement\n"
        
        if eval_data['precision'] >= 0.90:
            detail_content += "✓ High precision - Few false alarms\n"
        elif eval_data['precision'] >= 0.80:
            detail_content += "◐ Moderate precision - Some false alarms\n"
        else:
            detail_content += "✗ Low precision - Many false alarms\n"
        
        if eval_data['recall'] >= 0.90:
            detail_content += "✓ High recall - Few missed anomalies\n"
        elif eval_data['recall'] >= 0.80:
            detail_content += "◐ Moderate recall - Some missed anomalies\n"
        else:
            detail_content += "✗ Low recall - Many missed anomalies\n"
        
        if eval_data['f1_score'] >= 0.90:
            detail_content += "✓ Excellent balanced performance\n"
        elif eval_data['f1_score'] >= 0.80:
            detail_content += "◐ Good balanced performance\n"
        else:
            detail_content += "✗ Imbalanced or low performance\n"
        
        detail_content += "\n" + "="*70
        
        detail_text.insert(1.0, detail_content)
        detail_text.config(state='disabled')
        
        ttk.Button(detail_window, text="Close", command=detail_window.destroy).pack(pady=10)
    
    def show_comparison_chart(self):
        """Show comparison chart for all evaluated models"""
        self.comparison_chart_var.set('accuracy_comparison')
        self.generate_comparison_chart()
    
    def generate_comparison_chart(self):
        """Generate comparison chart based on selected type"""
        chart_type = self.comparison_chart_var.get()
        self.comparison_figure.clear()
        
        evaluated_models = {k: v for k, v in self.evaluation_data.items()}
        
        if not evaluated_models:
            ax = self.comparison_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No evaluated models available', 
                    ha='center', va='center', fontsize=14)
            self.comparison_canvas.draw()
            return
        
        try:
            if chart_type == 'accuracy_comparison':
                self.plot_accuracy_comparison(evaluated_models)
            elif chart_type == 'precision_recall':
                self.plot_precision_recall(evaluated_models)
            elif chart_type == 'error_rates':
                self.plot_error_rates(evaluated_models)
            elif chart_type == 'confusion_matrix':
                self.plot_confusion_matrices(evaluated_models)
            elif chart_type == 'roc_analysis':
                self.plot_roc_analysis(evaluated_models)
            
            self.comparison_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate chart:\n{str(e)}")
    
    def plot_accuracy_comparison(self, models):
        """Plot accuracy comparison chart"""
        ax = self.comparison_figure.add_subplot(111)
        
        model_names = list(models.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        x = np.arange(len(model_names))
        width = 0.2
        
        colors = ['#1976D2', '#2E7D32', '#F57C00', '#C62828']
        
        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            values = [models[name][metric] for name in model_names]
            offset = width * (i - 1.5)
            bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        self.comparison_figure.tight_layout()
    
    def plot_precision_recall(self, models):
        """Plot precision-recall scatter chart"""
        ax = self.comparison_figure.add_subplot(111)
        
        precisions = [data['precision'] for data in models.values()]
        recalls = [data['recall'] for data in models.values()]
        f1_scores = [data['f1_score'] for data in models.values()]
        
        scatter = ax.scatter(recalls, precisions, c=f1_scores, s=200, 
                            cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=2)
        
        for name, recall, precision in zip(models.keys(), recalls, precisions):
            ax.annotate(name, (recall, precision), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        
        # Add diagonal line (F1 contours would be ideal)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        
        cbar = self.comparison_figure.colorbar(scatter, ax=ax)
        cbar.set_label('F1 Score', rotation=270, labelpad=20, fontweight='bold')
        
        self.comparison_figure.tight_layout()
    
    def plot_error_rates(self, models):
        """Plot error rates comparison"""
        ax = self.comparison_figure.add_subplot(111)
        
        model_names = list(models.keys())
        fpr = [models[name]['fpr'] for name in model_names]
        fnr = [models[name]['fnr'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, fpr, width, label='False Positive Rate', 
                       color='#F57C00', alpha=0.8)
        bars2 = ax.bar(x + width/2, fnr, width, label='False Negative Rate', 
                       color='#C62828', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax.set_title('Error Rates Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        self.comparison_figure.tight_layout()
    
    def plot_confusion_matrices(self, models):
        """Plot confusion matrices for all models"""
        n_models = len(models)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        for idx, (name, data) in enumerate(models.items()):
            ax = self.comparison_figure.add_subplot(rows, cols, idx + 1)
            
            cm = np.array([
                [data['true_positive'], data['false_negative']],
                [data['false_positive'], data['true_negative']]
            ])
            
            im = ax.imshow(cm, cmap='RdYlGn', alpha=0.8)
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, cm[i, j],
                                 ha="center", va="center", color="black",
                                 fontsize=12, fontweight='bold')
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Anomaly', 'Normal'])
            ax.set_yticklabels(['Anomaly', 'Normal'])
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)
            ax.set_title(f'{name}\nAcc: {data["accuracy"]:.3f}', fontsize=10, fontweight='bold')
        
        self.comparison_figure.suptitle('Confusion Matrices Comparison', 
                                        fontsize=14, fontweight='bold')
        self.comparison_figure.tight_layout()
    
    def plot_roc_analysis(self, models):
        """Plot ROC-like analysis"""
        ax = self.comparison_figure.add_subplot(111)
        
        for name, data in models.items():
            tpr = data['recall']  # True Positive Rate
            fpr = data['fpr']     # False Positive Rate
            
            ax.scatter(fpr, tpr, s=150, label=name, alpha=0.7, edgecolors='black', linewidth=2)
            ax.annotate(name, (fpr, tpr), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9)
        
        # Add diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Random Classifier')
        
        # Ideal point
        ax.scatter([0], [1], s=200, c='gold', marker='*', 
                  edgecolors='black', linewidth=2, label='Ideal', zorder=5)
        
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
        ax.set_title('ROC-like Analysis\n(Closer to top-left is better)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        self.comparison_figure.tight_layout()
    
    def save_comparison_chart(self):
        """Save comparison chart"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf")],
            initialfile=f"evaluation_comparison_{datetime.now().strftime('%y%m%d_%H%M')}"
        )
        
        if file_path:
            self.comparison_figure.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Chart saved:\n{file_path}")
    
    def export_evaluation_report(self):
        """Export comprehensive evaluation report"""
        if not self.evaluation_data:
            messagebox.showwarning("Warning", "No evaluation data to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")],
            initialfile=f"evaluation_report_{datetime.now().strftime('%y%m%d_%H%M')}"
        )
        
        if not file_path:
            return
        
        try:
            # Prepare data
            report_data = []
            for model_name, eval_data in self.evaluation_data.items():
                report_data.append({
                    'Model': model_name,
                    'Evaluation_Date': eval_data['timestamp'],
                    'Total_Samples': eval_data['total_samples'],
                    'Anomaly_Samples': eval_data['total_anomaly'],
                    'Normal_Samples': eval_data['total_normal'],
                    'True_Positive': eval_data['true_positive'],
                    'False_Negative': eval_data['false_negative'],
                    'True_Negative': eval_data['true_negative'],
                    'False_Positive': eval_data['false_positive'],
                    'Accuracy': eval_data['accuracy'],
                    'Precision': eval_data['precision'],
                    'Recall': eval_data['recall'],
                    'F1_Score': eval_data['f1_score'],
                    'Specificity': eval_data['specificity'],
                    'FPR': eval_data['fpr'],
                    'FNR': eval_data['fnr']
                })
            
            df = pd.DataFrame(report_data)
            
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            else:
                df.to_excel(file_path, index=False, engine='openpyxl')
            
            messagebox.showinfo("Success", f"Evaluation report exported:\n{file_path}\n\nTotal models: {len(report_data)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report:\n{str(e)}")        
    
    def export_models_to_csv(self):
        """모든 모델 정보를 통합하여 CSV 저장 (loaded_models_info 기반)"""
        if not self.loaded_models_info:
            messagebox.showwarning("Warning", "No models to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"lstmif_models_full_{datetime.now().strftime('%y%m%d_%H%M')}.csv"
        )
        
        if not file_path:
            return
        
        try:
            export_data = []
            
            for timestamp, model_info in sorted(self.loaded_models_info.items(), 
                                               key=lambda x: x[0], reverse=True):
                metadata = model_info['metadata']
                
                # ========== 기본 정보 추출 ==========
                train_stats = metadata.get('train_stats', {})
                val_stats = metadata.get('val_stats', {})
                params = metadata.get('params', {})
                training_params = metadata.get('training_params', {})
                history = metadata.get('history', {})
                
                # ========== 계산된 메트릭 ==========
                train_mse = train_stats.get('mean', 0)
                val_mse = val_stats.get('mean', 0)
                
                # Final Loss
                train_loss_final = history['loss'][-1] if history.get('loss') else train_mse
                val_loss_final = history['val_loss'][-1] if history.get('val_loss') else val_mse
                
                # Val/Train Ratio
                val_train_ratio = val_loss_final / train_loss_final if train_loss_final > 0 else 0
                
                # Val Loss Std
                if history.get('val_loss') and len(history['val_loss']) >= 20:
                    val_loss_std = np.std(history['val_loss'][-20:])
                else:
                    val_loss_std = 0
                
                # Converged
                best_epoch = metadata.get('best_epoch', 0)
                epochs_trained = training_params.get('epochs_trained', 0)
                early_gap = epochs_trained - best_epoch
                converged = early_gap <= training_params.get('patience', 20) + 5
                
                # MAE
                train_mae = history['mae'][-1] if history.get('mae') else train_mse ** 0.5
                val_mae = history['val_mae'][-1] if history.get('val_mae') else val_mse ** 0.5
                
                # Q95
                train_q95 = train_stats.get('q95', train_mse * 1.5)
                val_q95 = val_stats.get('q95', val_mse * 1.5)
                
                # Status
                if val_loss_final < 0.005 and val_train_ratio < 1.8 and converged:
                    status = "Excellent"
                elif val_loss_final < 0.008 and val_train_ratio < 2.0 and converged:
                    status = "Good"
                elif val_loss_final < 0.015 and val_train_ratio < 2.5:
                    status = "Warning"
                else:
                    status = "Poor"
                
                # Duration
                train_time = metadata.get('training_time', 0)
                if train_time >= 3600:
                    duration_str = f"{train_time/3600:.1f}h"
                elif train_time >= 60:
                    duration_str = f"{train_time/60:.1f}m"
                else:
                    duration_str = f"{train_time:.0f}s"
                
                # Ensemble 확인
                model_id = metadata.get('model_id', 'unknown')
                ensemble_path = self.training_data_dir / "LSTM_IF" / f"ensemble_{model_id}.pkl"
                ensemble = "Yes" if ensemble_path.exists() else "No"
                
                # 아키텍처 - 모든 변수를 한 번에 추출
                encoder_units = params.get('encoder_units', [64, 32])
                latent_dim = params.get('latent_dim', 16)
                dropout = params.get('dropout', 0.25)
                recurrent_dropout = params.get('recurrent_dropout', 0.15)
                lr = params.get('lr', 0.001)
                optimizer = params.get('optimizer', 'adam')
                latent_activation = params.get('latent_activation', 'tanh')
                l2_reg = params.get('l2_reg', 0.001)
                clip_norm = params.get('clip_norm', 1.0)
                use_bidir = params.get('use_bidirectional', True)
                use_batch_norm = params.get('use_batch_norm', True)
                
                # Training Config
                batch_size = training_params.get('batch_size', 'N/A')
                total_epochs = training_params.get('epochs', 'N/A')
                augmentation = training_params.get('augmentation', 'N/A')
                patience = training_params.get('patience', 'N/A')
                min_delta = training_params.get('min_delta', 'N/A')
                val_split = training_params.get('validation_split', 'N/A')
                lr_schedule = training_params.get('lr_schedule', 'N/A')
                save_checkpoints = training_params.get('save_checkpoints', 'N/A')
                
                # ========== CSV Row 구성 ==========
                row = {
                    # Overview
                    'Timestamp': timestamp,
                    'Status': status,
                    'Val_Loss': f"{val_loss_final:.6f}",
                    'Val_Train_Ratio': f"{val_train_ratio:.2f}",
                    'Converged': 'Yes' if converged else 'No',
                    'Threshold': f"{metadata.get('threshold', 0):.6f}",
                    'Best_Epoch': f"{best_epoch}/{epochs_trained}",
                    'Duration': duration_str,
                    'Ensemble': ensemble,
                    
                    # Metrics
                    'Train_Loss': f"{train_loss_final:.6f}",
                    'Train_MAE': f"{train_mae:.6f}",
                    'Val_MAE': f"{val_mae:.6f}",
                    'Train_MSE': f"{train_mse:.6f}",
                    'Val_MSE': f"{val_mse:.6f}",
                    'Val_Std': f"{val_loss_std:.6f}",
                    'Train_Q95': f"{train_q95:.6f}",
                    'Val_Q95': f"{val_q95:.6f}",
                    
                    # Architecture
                    'Encoder_L1': encoder_units[0] if len(encoder_units) > 0 else 'N/A',
                    'Encoder_L2': encoder_units[1] if len(encoder_units) > 1 else 'N/A',
                    'Latent_Dim': latent_dim,
                    'Decoder_L1': encoder_units[1] if len(encoder_units) > 1 else 'N/A',
                    'Decoder_L2': encoder_units[0] if len(encoder_units) > 0 else 'N/A',
                    'Dropout': f"{dropout:.2f}",
                    'Recurrent_Dropout': f"{recurrent_dropout:.2f}",
                    'Learning_Rate': f"{lr:.6f}",
                    'Optimizer': optimizer,
                    'Latent_Activation': latent_activation,
                    'L2_Regularization': f"{l2_reg:.4f}",
                    'Gradient_Clip_Norm': f"{clip_norm:.2f}",
                    'Bidirectional': 'Yes' if use_bidir else 'No',
                    'Batch_Normalization': 'Yes' if use_batch_norm else 'No',
                    
                    # Training Config
                    'Batch_Size': batch_size,
                    'Total_Epochs': total_epochs,
                    'Trained_Epochs': epochs_trained,
                    'Augmentation': augmentation,
                    'Patience': patience,
                    'Min_Delta': f"{min_delta:.6f}" if isinstance(min_delta, (int, float)) else min_delta,
                    'Val_Split': f"{val_split:.2f}" if isinstance(val_split, (int, float)) else val_split,
                    'LR_Schedule': lr_schedule,
                    'Save_Checkpoint': 'Yes' if save_checkpoints is True else ('No' if save_checkpoints is False else 'N/A')
                }
                
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            self.log(f"✅ Exported {len(export_data)} models to CSV (Full data)")
            messagebox.showinfo("Success", 
                              f"Models exported successfully:\n{file_path}\n\n"
                              f"Total models: {len(export_data)}\n"
                              f"All data included (Overview + Metrics + Architecture + Config)")
            
        except Exception as e:
            self.log(f"❌ Export failed: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to export models:\n{str(e)}")    
    
    def load_existing_models(self):
        """모든 Sub-Tab에 모델 정보 로드 + 통계 업데이트"""
        # 기존 데이터 클리어
        for tree in [self.model_tree_overview, self.model_tree_metrics, 
                     self.model_tree_arch, self.model_tree_config]:
            tree.delete(*tree.get_children())
        
        self.loaded_models_info = {}
        
        if not self.training_data_dir:
            self._update_quick_stats(0, 0, 0, 0, 0, None)
            return
        
        models_dir = self.training_data_dir / "LSTM_IF"
        if not models_dir.exists():
            self._update_quick_stats(0, 0, 0, 0, 0, None)
            return
        
        model_files = list(models_dir.glob("lstmae_only_*.h5")) + list(models_dir.glob("lstmif_ensemble_*.h5"))
        
        # 통계 변수
        total = len(model_files)
        count_excellent = 0
        count_good = 0
        count_warning = 0
        count_poor = 0
        best_val_loss = float('inf')
        
        for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
            metadata_path = model_file.parent / f"{model_file.stem}_metadata.pkl"
            
            if not metadata_path.exists():
                continue
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            parts = model_file.stem.split('_')
            timestamp = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 4 else 'unknown'
            
            train_stats = metadata.get('train_stats', {})
            val_stats = metadata.get('val_stats', {})
            params = metadata.get('params', {})
            training_params = metadata.get('training_params', {})
            history = metadata.get('history', {})
            
            self.loaded_models_info[timestamp] = {
                'path': model_file,
                'metadata': metadata
            }
            
            # ========== 계산된 메트릭 ==========
            train_mse = train_stats.get('mean', 0)
            val_mse = val_stats.get('mean', 0)
            
            # Final Loss (history의 마지막 값)
            train_loss_final = history['loss'][-1] if history.get('loss') else train_mse
            val_loss_final = history['val_loss'][-1] if history.get('val_loss') else val_mse
            
            # Val/Train Ratio
            val_train_ratio = val_loss_final / train_loss_final if train_loss_final > 0 else 0
            
            # Val Loss Std (최근 20 epoch)
            if history.get('val_loss') and len(history['val_loss']) >= 20:
                val_loss_std = np.std(history['val_loss'][-20:])
            else:
                val_loss_std = 0
            
            # Converged (Early stopping 정상 작동 여부)
            best_epoch = metadata.get('best_epoch', 0)
            epochs_trained = training_params.get('epochs_trained', 0)
            early_gap = epochs_trained - best_epoch
            converged = early_gap <= training_params.get('patience', 20) + 5
            
            # MAE
            train_mae = history['mae'][-1] if history.get('mae') else train_mse ** 0.5
            val_mae = history['val_mae'][-1] if history.get('val_mae') else val_mse ** 0.5
            
            # Q95 (Percentile)
            train_q95 = train_stats.get('q95', train_mse * 1.5)
            val_q95 = val_stats.get('q95', val_mse * 1.5)
            
            # Status 계산
            if val_loss_final < 0.005 and val_train_ratio < 1.8 and converged:
                status = "🟢 Excellent"
                status_tag = 'excellent'
                count_excellent += 1
            elif val_loss_final < 0.008 and val_train_ratio < 2.0 and converged:
                status = "🟢 Good"
                status_tag = 'good'
                count_good += 1
            elif val_loss_final < 0.015 and val_train_ratio < 2.5:
                status = "🟡 Warning"
                status_tag = 'warning'
                count_warning += 1
            else:
                status = "🔴 Poor"
                status_tag = 'poor'
                count_poor += 1
            
            # Best model 추적
            if val_loss_final < best_val_loss:
                best_val_loss = val_loss_final
            
            # Duration
            train_time = metadata.get('training_time', 0)
            if train_time >= 3600:
                duration_str = f"{train_time/3600:.1f}h"
            elif train_time >= 60:
                duration_str = f"{train_time/60:.1f}m"
            else:
                duration_str = f"{train_time:.0f}s"
            
            # Ensemble 확인
            ensemble_path = model_file.parent / f"ensemble_{metadata.get('model_id', 'unknown')}.pkl"
            ensemble = "✅" if ensemble_path.exists() else "❌"
            
            # ✅ Evaluation 확인
            eval_info = metadata.get('evaluation', None)
            if eval_info:
                accuracy = eval_info.get('accuracy', 0)
                precision = eval_info.get('precision', 0)
                recall = eval_info.get('recall', 0)
                f1 = eval_info.get('f1_score', 0)
                eval_text = f"✅ F1:{f1:.3f}"
            else:
                eval_text = "❌ No Eval"
            
            # ========== 아키텍처 파라미터 추출 =========
            encoder_units = params.get('encoder_units', [64, 32])
            latent_dim = params.get('latent_dim', 16)
            dropout = params.get('dropout', 0.25)
            recurrent_dropout = params.get('recurrent_dropout', 0.15)
            lr = params.get('lr', 0.001)
            l2_reg = params.get('l2_reg', 0.001)
            clip_norm = params.get('clip_norm', 1.0)
            use_bidir = params.get('use_bidirectional', True)
            optimizer = params.get('optimizer', 'adam')
            latent_activation = params.get('latent_activation', 'tanh')
            use_batch_norm = params.get('use_batch_norm', True)           
            
            # 학습 설정
            batch_size = training_params.get('batch_size', 'N/A')
            total_epochs = training_params.get('epochs', 'N/A')
            augmentation = training_params.get('augmentation', 'N/A')
            patience = training_params.get('patience', 'N/A')
            min_delta = training_params.get('min_delta', 'N/A')
            val_split = training_params.get('validation_split', 'N/A')
            lr_schedule = training_params.get('lr_schedule', 'N/A')
            save_checkpoints = training_params.get('save_checkpoints', 'N/A')
            
            # 포맷팅
            min_delta_str = f"{min_delta:.4f}" if isinstance(min_delta, (int, float)) else min_delta
            val_split_str = f"{val_split:.2f}" if isinstance(val_split, (int, float)) else val_split
            save_ckpt_str = 'Y' if save_checkpoints is True else ('N' if save_checkpoints is False else 'N/A')
            converged_str = '✅' if converged else '❌'
            
            # ========== Overview 탭 ==========
            self.model_tree_overview.insert('', 'end', iid=timestamp, values=(
                timestamp,
                status,
                f"{val_loss_final:.6f}",
                f"{val_train_ratio:.2f}",
                converged_str,
                f"{metadata.get('threshold', 0):.5f}",
                f"{best_epoch}/{epochs_trained}",
                duration_str,
                ensemble,
                eval_text  # ✅ Evaluation 정보 추가
            ), tags=(status_tag,))
            
            # ========== Metrics 탭 ==========
            self.model_tree_metrics.insert('', 'end', iid=timestamp, values=(
                timestamp,
                f"{train_loss_final:.6f}",
                f"{val_loss_final:.6f}",
                f"{val_train_ratio:.2f}",
                f"{train_mae:.6f}",
                f"{val_mae:.6f}",
                f"{train_mse:.6f}",
                f"{val_mse:.6f}",
                f"{val_loss_std:.6f}",
                f"{train_q95:.6f}",
                f"{val_q95:.6f}",
                f"{metadata.get('threshold', 0):.5f}"
            ))
            
            # ========== Architecture 탭 ==========
            self.model_tree_arch.insert('', 'end', iid=timestamp, values=(
                timestamp,
                encoder_units[0] if len(encoder_units) > 0 else 'N/A',
                encoder_units[1] if len(encoder_units) > 1 else 'N/A',
                latent_dim,
                encoder_units[1] if len(encoder_units) > 1 else 'N/A',  # Decoder L1
                encoder_units[0] if len(encoder_units) > 0 else 'N/A',  # Decoder L2
                f"{dropout:.2f}",
                f"{recurrent_dropout:.2f}",
                f"{lr:.4f}",
                optimizer,  # ✅ 추가
                latent_activation,  # ✅ 추가
                f"{l2_reg:.4f}",
                f"{clip_norm:.2f}",
                'Y' if use_bidir else 'N',
                'Y' if use_batch_norm else 'N'  # ✅ 추가
            ))
            
            # ========== Training Config 탭 ==========
            self.model_tree_config.insert('', 'end', iid=timestamp, values=(
                timestamp,
                batch_size,
                total_epochs,
                epochs_trained,
                augmentation,
                patience,
                min_delta_str,
                val_split_str,
                lr_schedule,
                save_ckpt_str
            ))
        
        # 통계 업데이트
        self._update_quick_stats(total, count_excellent, count_good, count_warning, count_poor, best_val_loss)
        
        self.log(f"Loaded {len(model_files)} models from LSTM_IF folder")
    
    def _update_quick_stats(self, total, excellent, good, warning, poor, best_val_loss):
        """Quick Statistics 업데이트"""
        self.stats_labels['total'].config(text=f"Total: {total}")
        self.stats_labels['excellent'].config(text=f"🟢 Excellent: {excellent}")
        self.stats_labels['good'].config(text=f"🟢 Good: {good}")
        self.stats_labels['warning'].config(text=f"🟡 Warning: {warning}")
        self.stats_labels['poor'].config(text=f"🔴 Poor: {poor}")
        
        if best_val_loss and best_val_loss != float('inf'):
            self.stats_labels['best'].config(text=f"Best Val Loss: {best_val_loss:.6f}")
        else:
            self.stats_labels['best'].config(text="Best Val Loss: N/A")
            
    def sort_models_by_tab(self, column, tab_name):
        """각 Sub-Tab의 정렬 기능"""
        tree_map = {
            'overview': self.model_tree_overview,
            'metrics': self.model_tree_metrics,
            'arch': self.model_tree_arch,
            'config': self.model_tree_config
        }
        
        tree = tree_map.get(tab_name)
        if not tree:
            return
        
        items = [(tree.set(item, column), item) for item in tree.get_children('')]
        
        try:
            # 숫자 정렬 시도
            items.sort(key=lambda x: float(
                x[0].replace('s', '').replace('m', '').replace('h', '')
                     .replace('Y', '1').replace('N', '0').replace('N/A', '-1')
                     .replace('✅', '1').replace('❌', '0')
                     .replace('🟢 Excellent', '4').replace('🟢 Good', '3')
                     .replace('🟡 Warning', '2').replace('🔴 Poor', '1')
                     .replace('/', '.')
            ))
        except:
            # 문자열 정렬
            items.sort()
        
        for index, (val, item) in enumerate(items):
            tree.move(item, '', index)        
                
    def show_model_details(self):
        """Show detailed information of selected model in a popup"""
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to view details.")
            return
        
        timestamp = selection[0]
        if timestamp not in self.loaded_models_info:
            return
        
        metadata = self.loaded_models_info[timestamp]['metadata']
        params = metadata.get('params', {})
        training_params = metadata.get('training_params', {})
        train_stats = metadata.get('train_stats', {})
        val_stats = metadata.get('val_stats', {})
        
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Model Details - {timestamp}")
        detail_window.geometry("700x600")
        
        detail_notebook = ttk.Notebook(detail_window)
        detail_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        config_frame = ttk.Frame(detail_notebook)
        detail_notebook.add(config_frame, text="Configuration")
        
        config_text = scrolledtext.ScrolledText(config_frame, wrap=tk.WORD, width=80, height=25)
        config_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 안전한 문자열 변환 함수
        def safe_str(value, default='N/A'):
            if value is None or value == 'N/A':
                return default
            return str(value)
        
        # 기본 정보 구성
        config_lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║                    MODEL CONFIGURATION                        ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            "📋 Model Information",
            f"  • Model ID: {safe_str(metadata.get('model_id'))}",
            f"  • Created: {safe_str(metadata.get('created_at'))}",
            f"  • Training Duration: {metadata.get('training_time', 0):.1f}s",
            f"  • Best Epoch: {safe_str(metadata.get('best_epoch'))}",
            "",
            "🏗 Architecture Parameters",
            f"  • Encoder Layer 1: {safe_str(params.get('encoder_units', [64, 32])[0] if params.get('encoder_units') else None)}",
            f"  • Encoder Layer 2: {safe_str(params.get('encoder_units', [64, 32])[1] if len(params.get('encoder_units', [])) > 1 else None)}",
            f"  • Latent Dimension: {safe_str(params.get('latent_dim'))}",
            f"  • Dropout: {safe_str(params.get('dropout'))}",
            f"  • Recurrent Dropout: {safe_str(params.get('recurrent_dropout'))}",
            f"  • L2 Regularization: {safe_str(params.get('l2_reg'))}",
            f"  • Gradient Clip Norm: {safe_str(params.get('clip_norm'))}",
            f"  • Bidirectional: {'Yes' if params.get('use_bidirectional', True) else 'No'}",
            "",
            "⚙️ Training Parameters",
            f"  • Learning Rate: {safe_str(params.get('lr'))}",
            f"  • Batch Size: {safe_str(training_params.get('batch_size'))}",
            f"  • Total Epochs (Config): {safe_str(training_params.get('epochs'))}",
            f"  • Epochs Trained (Actual): {safe_str(training_params.get('epochs_trained'))}",
            f"  • Augmentation Factor: {safe_str(training_params.get('augmentation'))}",
            f"  • Patience: {safe_str(training_params.get('patience'))}",
            f"  • Min Delta: {safe_str(training_params.get('min_delta'))}",
            f"  • Validation Split: {safe_str(training_params.get('validation_split'))}",
            f"  • LR Schedule: {safe_str(training_params.get('lr_schedule'))}",
            f"  • Save Checkpoints: {safe_str(training_params.get('save_checkpoints'))}",
            f"  • Train Samples: {safe_str(training_params.get('train_samples'))}",
            f"  • Val Samples: {safe_str(training_params.get('val_samples'))}",
            "",
            "🎯 Threshold Settings",
            f"  • Base Threshold: {metadata.get('threshold', 0):.6f}",
            "",
            "  Range-Specific Adjustments:"
        ]
        
        # Range adjustments 추가
        for range_name, adj in sorted(metadata.get('vacuum_range_thresholds', {}).items()):
            adjusted_threshold = metadata.get('threshold', 0) * adj
            config_lines.append(f"    • {range_name}: {adj:.3f}x → {adjusted_threshold:.6f}")
        
        config_info = '\n'.join(config_lines)
        config_text.insert('1.0', config_info)
        config_text.config(state='disabled')
        
        stats_frame = ttk.Frame(detail_notebook)
        detail_notebook.add(stats_frame, text="Statistics")
        
        stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, width=80, height=25)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        stats_lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║                    PERFORMANCE STATISTICS                     ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            "📊 Training Set Statistics",
            f"  • Mean MSE: {train_stats.get('mean', 0):.6f}",
            f"  • Std Dev: {train_stats.get('std', 0):.6f}",
            f"  • Min: {train_stats.get('min', 0):.6f}",
            f"  • Max: {train_stats.get('max', 0):.6f}",
            f"  • Q25: {train_stats.get('q25', 0):.6f}",
            f"  • Q50 (Median): {train_stats.get('q50', 0):.6f}",
            f"  • Q75: {train_stats.get('q75', 0):.6f}",
            f"  • Q95: {train_stats.get('q95', 0):.6f}",
            f"  • Q99: {train_stats.get('q99', 0):.6f}",
            "",
            "📈 Validation Set Statistics",
            f"  • Mean MSE: {val_stats.get('mean', 0):.6f}",
            f"  • Std Dev: {val_stats.get('std', 0):.6f}",
            f"  • Min: {val_stats.get('min', 0):.6f}",
            f"  • Max: {val_stats.get('max', 0):.6f}",
            f"  • Q25: {val_stats.get('q25', 0):.6f}",
            f"  • Q50 (Median): {val_stats.get('q50', 0):.6f}",
            f"  • Q75: {val_stats.get('q75', 0):.6f}",
            f"  • Q95: {val_stats.get('q95', 0):.6f}",
            f"  • Q99: {val_stats.get('q99', 0):.6f}",
            "",
            "📉 Training History"
        ]
        
        # History 정보 안전하게 추가
        history = metadata.get('history', {})
        if history.get('loss'):
            stats_lines.append(f"  • Final Train Loss: {history['loss'][-1]:.6f}")
        else:
            stats_lines.append("  • Final Train Loss: N/A")
        
        if history.get('val_loss'):
            stats_lines.append(f"  • Final Val Loss: {history['val_loss'][-1]:.6f}")
            stats_lines.append(f"  • Best Val Loss: {min(history['val_loss']):.6f}")
        else:
            stats_lines.append("  • Final Val Loss: N/A")
            stats_lines.append("  • Best Val Loss: N/A")
        
        stats_info = '\n'.join(stats_lines)
        stats_text.insert('1.0', stats_info)
        stats_text.config(state='disabled')
        
        ttk.Button(detail_window, text="Close", command=detail_window.destroy).pack(pady=10)
    
    def sort_models_by(self, column):
        """Sort models by selected column"""
        items = [(self.model_tree.set(item, column), item) for item in self.model_tree.get_children('')]
        
        try:
            items.sort(key=lambda x: float(x[0].replace('s', '').replace('m', '').replace('h', '').replace('Y', '1').replace('N', '0').replace('N/A', '-1')))
        except:
            items.sort()
        
        for index, (val, item) in enumerate(items):
            self.model_tree.move(item, '', index)             
    
    def select_train_folder(self):
        folder = filedialog.askdirectory(title="Select Training Data Folder")
        if folder:
            self.train_path_var.set(folder)
            self.training_data_dir = Path(folder)
            self.load_existing_models()
            self.refresh_model_list()
    
    def select_test_folder(self):
        folder = filedialog.askdirectory(title="Select Test Data Folder")
        if folder:
            self.test_path_var.set(folder)
    
    def start_training(self):
        if not self.train_path_var.get():
            messagebox.showerror("Error", "Please select training data folder.")
            return
        
        self.stop_training_event.clear()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        thread = Thread(target=self.train_model, daemon=True)
        thread.start()
    
    def stop_training(self):
        self.stop_training_event.set()
        self.log("⏹ Training stop requested...")
        self.stop_btn.config(state='disabled')
    
    def epoch_progress_callback(self, epoch, logs, progress, phase='end', batch=None):
        """Enhanced callback for epoch progress with detailed logging"""
        
        if phase == 'begin':
            # Epoch 시작
            self.epoch_log.insert(tk.END, f"\n{'='*70}\n", 'epoch_start')
            self.epoch_log.insert(tk.END, f"Epoch {epoch} Started\n", 'epoch_start')
            self.epoch_log.insert(tk.END, f"{'='*70}\n", 'epoch_start')
            self.epoch_log.see(tk.END)
            
        elif phase == 'end':
            # Epoch 종료 - 진행률 업데이트
            self.progress_var.set(progress)
            self.progress_label.config(
                text=f"Epoch {epoch}/{self.param_vars['epochs'].get()}: "
                     f"Loss={logs.get('loss', 0):.6f}, Val Loss={logs.get('val_loss', 0):.6f}"
            )
            
            # Epoch Log에 상세 정보 기록
            self.epoch_log.insert(tk.END, f"\nEpoch {epoch} Completed:\n", 'epoch_end')
            
            # 학습 메트릭
            train_loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            train_mae = logs.get('mae', 0)
            train_mse = logs.get('mse', 0)
            val_mae = logs.get('val_mae', 0)
            val_mse = logs.get('val_mse', 0)
            
            self.epoch_log.insert(tk.END, f"  Train Loss: {train_loss:.6f}\n")
            self.epoch_log.insert(tk.END, f"  Val Loss:   {val_loss:.6f}\n")
            self.epoch_log.insert(tk.END, f"  Train MAE:  {train_mae:.6f} | MSE: {train_mse:.6f}\n")
            self.epoch_log.insert(tk.END, f"  Val MAE:    {val_mae:.6f} | MSE: {val_mse:.6f}\n")
            
            # 이전 epoch와 비교하여 개선 여부 표시
            if hasattr(self, '_last_val_loss'):
                improvement = self._last_val_loss - val_loss
                if improvement > 0:
                    self.epoch_log.insert(tk.END, 
                        f"  ✓ Improved: {improvement:.6f} ({improvement/self._last_val_loss*100:.2f}%)\n", 
                        'loss_improved')
                else:
                    self.epoch_log.insert(tk.END, 
                        f"  ✗ Degraded: {-improvement:.6f} ({-improvement/self._last_val_loss*100:.2f}%)\n", 
                        'loss_worse')
            
            self._last_val_loss = val_loss
            
            # Learning rate 정보
            if 'lr' in logs:
                self.epoch_log.insert(tk.END, f"  Learning Rate: {logs['lr']:.6e}\n", 'lr_change')
            
            # 시간 정보 (있는 경우)
            if hasattr(self, '_epoch_start_time'):
                epoch_duration = time.time() - self._epoch_start_time
                self.epoch_log.insert(tk.END, f"  Duration: {epoch_duration:.1f}s\n")
            
            self._epoch_start_time = time.time()
            
            self.epoch_log.insert(tk.END, f"{'-'*70}\n")
            self.epoch_log.see(tk.END)
            
        elif phase == 'batch' and batch is not None:
            # 배치 진행 상황 (선택적, 너무 많으면 성능 저하 가능)
            if batch % 50 == 0:  # 50 배치마다 업데이트
                self.epoch_log.insert(tk.END, f"  Batch {batch}: ", 'epoch_start')
                if logs:
                    batch_loss = logs.get('loss', 0)
                    self.epoch_log.insert(tk.END, f"Loss={batch_loss:.6f}\n")
                else:
                    self.epoch_log.insert(tk.END, "Processing...\n")
        
        self.root.update_idletasks()
    
    def train_model(self):
        try:
            # 초기화
            self._last_val_loss = float('inf')
            self._epoch_start_time = time.time()
            
            # Epoch Log 초기화
            self.epoch_log.delete(1.0, tk.END)
            self.epoch_log.insert(tk.END, "="*70 + "\n")
            self.epoch_log.insert(tk.END, "LSTM-AE Training - Epoch Progress Monitor\n")
            self.epoch_log.insert(tk.END, "="*70 + "\n\n")
            
            folder_path = Path(self.train_path_var.get())
            all_data = self.load_all_data(folder_path)
            
            if len(all_data) == 0:
                self.log("❌ ERROR: No valid data found")
                return
            
            # 첫 번째 샘플로 데이터 검증
            sample = all_data[0]
            self.log(f"\n{'='*60}")
            self.log(f"📊 DATA VALIDATION")
            self.log(f"{'='*60}")
            self.log(f"✅ Total files loaded: {len(all_data)}")
            self.log(f"✅ Sample shape: {sample.shape}")
            
            # 원본 데이터 범위 확인
            self.log(f"\n📈 Raw Data Ranges:")
            self.log(f"  Vacuum: {sample[:, 0].min():.2e} ~ {sample[:, 0].max():.2e} Pa")
            self.log(f"  Temp:   {sample[:, 1:5].min():.2f} ~ {sample[:, 1:5].max():.2f} K")
            self.log(f"  Press:  {sample[:, 5:7].min():.2f} ~ {sample[:, 5:7].max():.2f} MPa")
            
            # 이상치 경고
            if sample[:, 0].max() > 0.01:
                self.log(f"⚠️ WARNING: Vacuum exceeds 0.01 Pa (clipping will occur)")
            if sample[:, 1:5].max() > 60:
                self.log(f"⚠️ WARNING: Temperature exceeds 60 K (clipping will occur)")
            if sample[:, 5:7].max() > 4.0:
                self.log(f"⚠️ WARNING: Pressure exceeds 4.0 MPa (clipping will occur)")
            
            # NaN/Inf 체크
            if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
                self.log(f"❌ ERROR: Data contains NaN or Inf values!")
                return
            
            # 정규화 테스트
            test_model = EnhancedLSTMIFModel()
            processed = test_model.preprocess_data(sample)
            
            self.log(f"\n🔧 After Preprocessing:")
            self.log(f"  Range: {processed.min():.2f} ~ {processed.max():.2f}")
            self.log(f"  Mean: {processed.mean():.3f}, Std: {processed.std():.3f}")
            
            if abs(processed.min()) > 10 or abs(processed.max()) > 10:
                self.log(f"❌ ERROR: Preprocessing failed! Values exceed ±10")
                self.log(f"→ Check Clipping bounds or data quality")
                messagebox.showerror("Error", 
                    "Data preprocessing failed!\nValues are out of expected range.\n"
                    "Please check the training log for details.")
                return
            
            self.log(f"✅ Data validation passed!")
            self.log(f"{'='*60}\n")
            
            self.log(f"\n{'='*60}")
            self.log(f"🎓 Training Enhanced LSTM-AE Model")
            self.log(f"📁 Total datasets: {len(all_data)}")
            
            vacuum_dist = {}
            for data in all_data:
                range_name = VacuumRangeManager.get_range_name(data[0, 0])
                vacuum_dist[range_name] = vacuum_dist.get(range_name, 0) + 1
            
            self.log("📊 Vacuum Range Distribution:")
            for range_name, count in sorted(vacuum_dist.items()):
                self.log(f"  {range_name}: {count} ({count/len(all_data)*100:.1f}%)")
            
            params = {k: v.get() for k, v in self.param_vars.items()}
            encoder_units = [params['enc0'], params['enc1']]
            
            model = EnhancedLSTMIFModel(
                encoder_units=encoder_units,
                latent_dim=params['latent'],
                decoder_units=encoder_units[::-1],
                dropout=params['dropout'],
                recurrent_dropout=params['rec_drop'],
                lr=params['lr'],
                use_bidirectional=params['bidir'],
                l2_reg=params['l2_reg'],
                clip_norm=params['clip_norm'],
                optimizer=params['optimizer'],
                latent_activation=params['latent_activation'],
                use_batch_norm=params['use_batch_norm']
            )
            
            split_idx = int(len(all_data) * (1 - params['val_split']))
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
            
            self.log(f"📚 Train: {len(train_data)}, Val: {len(val_data)}")
            
            output_dir = folder_path / "LSTM_IF"
            output_dir.mkdir(exist_ok=True)
            
            checkpoint_dir = output_dir / "checkpoints" if params['save_checkpoints'] else None
            if checkpoint_dir:
                checkpoint_dir.mkdir(exist_ok=True)
            
            history = model.train(
                train_data, val_data,
                epochs=params['epochs'],
                batch_size=params['batch'],
                augmentation=params['augment'],
                patience=params['patience'],
                min_delta=params['min_delta'],
                lr_schedule=params['lr_schedule'],
                save_checkpoints=params['save_checkpoints'],
                checkpoint_dir=checkpoint_dir,
                stop_event=self.stop_training_event,
                progress_callback=self.epoch_progress_callback,
                verbose=1
            )
            
            if self.stop_training_event.is_set():
                self.log("\n⚠ Training stopped by user")
            else:
                self.log(f"\n✅ LSTM-AE Training Completed!")
            
            self.log(f"🆔 Model ID: {model.model_id}")
            self.log(f"⏱ Training Time: {model.training_time:.1f}s")
            self.log(f"🏆 Best Epoch: {model.best_epoch}")
            self.log(f"🎯 Base Threshold: {model.threshold:.6f}")
            self.log(f"📊 Train MSE: {model.train_stats['mean']:.6f} ± {model.train_stats['std']:.6f}")
            self.log(f"📊 Val MSE: {model.val_stats['mean']:.6f} ± {model.val_stats['std']:.6f}")
            
            self.log("\n🔧 Range-specific adjustments:")
            for range_name, adj in sorted(model.vacuum_range_thresholds.items()):
                self.log(f"  {range_name}: {adj:.3f}x → {model.threshold * adj:.6f}")
            
            ensemble_enabled = self.param_vars['enable_ensemble'].get()
            saved_path = model.save(output_dir / "placeholder.h5", ensemble_enabled=ensemble_enabled)
            self.log(f"\n💾 Saved: {saved_path.name}")
    
            # Isolation Forest 학습 (if enabled)
            if self.param_vars['enable_ensemble'].get():
                self.log(f"\n{'='*60}")
                self.log(f"🌲 [PHASE 2/2] Starting Isolation Forest Training...")
                self.log(f"   This may take 1-3 minutes depending on data size...")
                self.log(f"{'='*60}")
                
                if_start_time = time.time()
                
                if_detector = IsolationForestDetector(
                    contamination=self.param_vars['if_contamination'].get(),
                    n_estimators=self.param_vars['if_n_estimators'].get()
                )
                
                # Train on same training data
                self.log(f"📊 Extracting features from {len(train_data)} samples...")
                if_detector.train(train_data)
                
                if_elapsed = time.time() - if_start_time
                self.log(f"✅ Isolation Forest Completed in {if_elapsed:.1f}s")
                self.log(f"   Features extracted: {if_detector.feature_scaler.n_features_in_}")
                self.log(f"   Estimators: {self.param_vars['if_n_estimators'].get()}")
                
                # Create ensemble
                self.ensemble_detector = EnsembleLSTMAE_IF(
                    lstmae_model=model,
                    if_detector=if_detector,
                    lstmae_weight=self.param_vars['lstmae_weight'].get(),
                    if_weight=self.param_vars['if_weight'].get(),
                    ensemble_strategy=self.param_vars['ensemble_strategy'].get()
                )
                
                self.log(f"✅ Ensemble created: {self.param_vars['ensemble_strategy'].get()}")
                self.log(f"   Weights - LSTM-AE: {self.param_vars['lstmae_weight'].get():.2f}, "
                        f"IF: {self.param_vars['if_weight'].get():.2f}")
                
                # Save IF detector and ensemble
                if_save_path = output_dir / f"isolation_forest_{model.model_id}.pkl"
                ensemble_save_path = output_dir / f"ensemble_{model.model_id}.pkl"
                
                with open(if_save_path, 'wb') as f:
                    pickle.dump(if_detector, f)
                
                with open(ensemble_save_path, 'wb') as f:
                    pickle.dump({
                        'strategy': self.param_vars['ensemble_strategy'].get(),
                        'lstmae_weight': self.param_vars['lstmae_weight'].get(),
                        'if_weight': self.param_vars['if_weight'].get()
                    }, f)
                
                self.log(f"💾 Saved IF model and ensemble config")
            
            # 최종 완료 메시지
            self.log(f"\n{'='*60}")
            if self.param_vars['enable_ensemble'].get():
                self.log(f"🎉 ALL TRAINING COMPLETED! (LSTM-AE + Isolation Forest)")
            else:
                self.log(f"🎉 TRAINING COMPLETED!")
            self.log(f"{'='*60}")
            
            self.model = model
            self.progress_var.set(100)
            
            # ✅ ========== 자동으로 모델 정보 새로고침 및 CSV 저장 ========== ✅
            self.log(f"\n{'='*60}")
            self.log(f"💾 AUTO-SAVING MODEL INFORMATION")
            self.log(f"{'='*60}")
            
            # 모델 목록 새로고침 (CSV 저장을 위해 필요)
            self.load_existing_models()
            
            # CSV 자동 저장
            csv_saved = self.auto_export_models_csv()
            
            if csv_saved:
                self.log(f"✅ Model information saved to CSV")
            else:
                self.log(f"⚠️ Model CSV save may have failed (check logs)")
            
            # 최종 성공 메시지
            success_message = "✅ Training completed successfully!\n\n"
            
            if self.param_vars['enable_ensemble'].get():
                success_message += "🔥 Ensemble Model (LSTM-AE + IF) trained\n\n"
            else:
                success_message += "🔍 LSTM-AE Model trained\n\n"
            
            success_message += (
                f"Model saved:\n"
                f"  • {saved_path.name}\n\n"
                f"Model information saved to:\n"
                f"  • LSTM_IF/reports/lstmif_models_full_*.csv"
            )
            
            messagebox.showinfo("Training Complete", success_message)
            
        except Exception as e:
            self.log(f"\n❌ ERROR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", str(e))
        
        finally:
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.stop_training_event.clear()

    def _extract_true_label(self, filename):
        """
        파일명에서 Ground Truth Label을 추출하는 함수
        
        Samsung 파일 명명 규칙에 맞게 커스터마이징 필요:
        - "S1_UNIT1_CH1_20241028_001_OK.csv" -> 0 (정상)
        - "S1_UNIT1_CH1_20241028_002_NG.csv" -> 1 (이상)
        
        현재는 키워드 기반으로 판단하며, 필요시 별도의 Label 파일 로딩으로 변경 가능
        """
        filename_lower = filename.lower()
        
        # NG, Anomaly, Error 키워드 → 이상 (1)
        if any(kw in filename_lower for kw in ['ng', 'anomaly', 'abnormal', 'error', 'fail', 'bad']):
            return 1
        
        # OK, Normal, Good 키워드 → 정상 (0)
        if any(kw in filename_lower for kw in ['ok', 'normal', 'good', 'pass']):
            return 0
        
        # 기본값: 0 (정상으로 가정)
        # ⚠️ 실제 환경에 맞게 수정 필요
        return 0            
    
    def run_detection(self):
        if not self.test_path_var.get():
            messagebox.showerror("Error", "Please select test folder.")
            return
        
        thread = Thread(target=self.perform_detection, daemon=True)
        thread.start()
    
    def perform_detection(self):
        """
        실제 Detection을 수행하고 결과를 표시하는 함수
        ✅ Evaluation 메트릭 계산 및 저장 포함
        """
        try:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, "="*70 + "\n")
            self.result_text.insert(tk.END, "🚀 STARTING ANOMALY DETECTION\n")
            self.result_text.insert(tk.END, "="*70 + "\n\n")
            
            # 1. 테스트 데이터 확인
            if not self.test_data_dir or not self.test_data_dir.exists():
                self.result_text.insert(tk.END, "❌ Test data folder not selected\n")
                messagebox.showerror("Error", "Please select test data folder")
                return
            
            csv_files = sorted(self.test_data_dir.glob("*.csv"))
            self.result_text.insert(tk.END, f"📂 Test Folder: {self.test_data_dir}\n")
            self.result_text.insert(tk.END, f"📄 Found {len(csv_files)} CSV files\n\n")
            self.result_text.update()
            
            if len(csv_files) == 0:
                messagebox.showerror("Error", "No CSV files found")
                return
            
            # 2. 모델 로딩 (기존 코드 유지)
            selected_model_name = self.selected_model_var.get()
            
            if not selected_model_name:
                if self.training_data_dir:
                    models_dir = self.training_data_dir / "LSTM_IF"
                    if not models_dir.exists():
                        messagebox.showerror("Error", "LSTM_IF folder not found")
                        return
                    
                    model_files = list(models_dir.glob("lstmae_only_*.h5")) + list(models_dir.glob("lstmif_ensemble_*.h5"))
                    if not model_files:
                        messagebox.showerror("Error", "No trained models found")
                        return
                    
                    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                    selected_model_name = latest_model.name
                    self.selected_model_var.set(selected_model_name)
                    self.result_text.insert(tk.END, f"ℹ️ Auto-selected: {selected_model_name}\n")
                else:
                    messagebox.showerror("Error", "Training data folder not selected")
                    return
            
            self.result_text.insert(tk.END, f"📂 Loading model: {selected_model_name}\n")
            self.result_text.update()
            
            # 모델 경로 확인
            models_dir = self.training_data_dir / "LSTM_IF"
            selected_model_path = models_dir / selected_model_name
            
            if not selected_model_path.exists():
                messagebox.showerror("Error", f"Model not found: {selected_model_name}")
                return
            
            # 모델 ID 추출
            try:
                model_id_from_filename = selected_model_name.replace('lstmae_only_', '').replace('lstmif_ensemble_', '').replace('.h5', '')
            except:
                model_id_from_filename = "unknown"
            
            # 모델 로딩
            need_reload = (
                self.model is None or 
                not hasattr(self.model, 'model_id') or
                self.model.model_id != model_id_from_filename
            )
            
            if need_reload:
                self.result_text.insert(tk.END, f"⏳ Loading model...\n")
                self.result_text.update()
                
                try:
                    self.model = EnhancedLSTMIFModel()
                    self.model.load(selected_model_path)
                    self.result_text.insert(tk.END, f"✅ Model loaded\n")
                    
                    if self.model.threshold is None or self.model.threshold <= 0:
                        raise ValueError(f"Invalid threshold: {self.model.threshold}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                    return
            else:
                self.result_text.insert(tk.END, f"✅ Using loaded model\n")
            
            # 모델 정보 출력
            self.result_text.insert(tk.END, f"\n{'='*70}\n")
            self.result_text.insert(tk.END, f"📋 MODEL INFORMATION\n")
            self.result_text.insert(tk.END, f"{'='*70}\n")
            self.result_text.insert(tk.END, f"Model ID: {self.model.model_id}\n")
            self.result_text.insert(tk.END, f"Base Threshold: {self.model.threshold:.6f}\n")
            self.result_text.insert(tk.END, f"Created: {self.model.created_at}\n")
            self.result_text.update()
            
            self.model_threshold_label.config(text=f"{self.model.threshold:.6f}", foreground='green')
            
            # Active Threshold
            active_threshold = self.get_active_threshold()
            if active_threshold is None or active_threshold <= 0:
                messagebox.showerror("Error", "Invalid threshold")
                return
            
            threshold_type = "Custom" if self.use_custom_threshold_var.get() else "Model Default"
            self.result_text.insert(tk.END, f"Active Threshold: {active_threshold:.6f} ({threshold_type})\n")
            
            self.active_threshold_label.config(
                text=f"{active_threshold:.6f} ({threshold_type})",
                foreground='red' if self.use_custom_threshold_var.get() else 'green'
            )
            
            # 3. Ensemble 확인 (기존 코드 유지)
            detection_method = self.detection_method.get()
            use_ensemble = detection_method == "Ensemble"
            
            if use_ensemble:
                if hasattr(self.model, 'if_model') and self.model.if_model is not None:
                    self.result_text.insert(tk.END, f"\n🤖 Ensemble Mode: LSTM-AE + Isolation Forest\n")
                else:
                    self.result_text.insert(tk.END, f"\n⚠️ Ensemble requested but IF not trained\n")
                    self.result_text.insert(tk.END, f"ℹ️ Using LSTM-AE only\n")
                    use_ensemble = False
            else:
                self.result_text.insert(tk.END, f"\n🔍 Detection Mode: LSTM-AE Only\n")
            
            # 4. Detection 실행
            self.result_text.insert(tk.END, f"\n{'='*70}\n")
            self.result_text.insert(tk.END, f"🔍 DETECTION PROGRESS\n")
            self.result_text.insert(tk.END, f"{'='*70}\n")
            self.result_text.update()
            
            self.current_results = []
            
            # ✅ NEW: Evaluation용 데이터 수집
            all_predictions = []
            all_true_labels = []
            all_anomaly_scores = []
            
            for idx, file_path in enumerate(csv_files, 1):
                try:
                    df = pd.read_csv(file_path)
                    
                    if len(df) < 7000:
                        self.result_text.insert(tk.END, f"⚠️ [{idx}/{len(csv_files)}] Skipped {file_path.name} ({len(df)} rows)\n")
                        continue
                    
                    data = df.iloc[:7000, 1:8].values.astype(np.float32)
                    vacuum_value = float(np.median(data[:, 0]))
                    vacuum_range = VacuumRangeManager.get_range_name(vacuum_value)
                    
                    # Detection 수행
                    if use_ensemble:
                        result = self.model.detect_anomaly_ensemble(data, vacuum_range)
                    else:
                        result = self.model.detect_anomaly(data, vacuum_range)
                    
                    result['file'] = file_path.name
                    result['vacuum_range'] = vacuum_range
                    result['vacuum_value'] = vacuum_value
                    
                    self.current_results.append(result)
                    
                    # ✅ NEW: Evaluation 데이터 수집
                    prediction = 1 if result['is_anomaly'] else 0
                    all_predictions.append(prediction)
                    
                    true_label = self._extract_true_label(file_path.name)
                    all_true_labels.append(true_label)
                    
                    anomaly_score = result.get('anomaly_score', result.get('ensemble_score', 0))
                    all_anomaly_scores.append(anomaly_score)
                    
                    status = "🔴 ANOMALY" if result['is_anomaly'] else "🟢 NORMAL"
                    self.result_text.insert(tk.END, f"[{idx}/{len(csv_files)}] {file_path.name}: {status}\n")
                    
                    if idx % 10 == 0:
                        self.result_text.update()
                    
                except Exception as e:
                    self.result_text.insert(tk.END, f"❌ [{idx}/{len(csv_files)}] Error: {str(e)}\n")
                    continue
            
            # 5. ✅ NEW: Evaluation 메트릭 계산
            self.result_text.insert(tk.END, f"\n{'='*70}\n")
            self.result_text.insert(tk.END, f"📊 EVALUATION METRICS\n")
            self.result_text.insert(tk.END, f"{'='*70}\n")
            
            if len(all_predictions) > 0 and len(all_true_labels) > 0:
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score,
                    f1_score, confusion_matrix, roc_auc_score
                )
                
                accuracy = accuracy_score(all_true_labels, all_predictions)
                precision = precision_score(all_true_labels, all_predictions, zero_division=0)
                recall = recall_score(all_true_labels, all_predictions, zero_division=0)
                f1 = f1_score(all_true_labels, all_predictions, zero_division=0)
                cm = confusion_matrix(all_true_labels, all_predictions)
                
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                try:
                    auc_roc = roc_auc_score(all_true_labels, all_anomaly_scores)
                except:
                    auc_roc = 0.0
                
                # ✅ 클래스 변수에 저장 (auto_generate_full_report에서 사용)
                self.last_evaluation_metrics = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'false_positive_rate': float(fpr),
                    'auc_roc': float(auc_roc),
                    'confusion_matrix': {
                        'true_negative': int(tn),
                        'false_positive': int(fp),
                        'false_negative': int(fn),
                        'true_positive': int(tp)
                    },
                    'total_samples': len(all_predictions),
                    'anomaly_count': sum(all_predictions),
                    'normal_count': len(all_predictions) - sum(all_predictions),
                    'detection_method': detection_method,
                    'model_id': self.model.model_id,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.result_text.insert(tk.END, f"Accuracy:  {accuracy:.4f}\n")
                self.result_text.insert(tk.END, f"Precision: {precision:.4f}\n")
                self.result_text.insert(tk.END, f"Recall:    {recall:.4f}\n")
                self.result_text.insert(tk.END, f"F1-Score:  {f1:.4f}\n")
                self.result_text.insert(tk.END, f"FPR:       {fpr:.4f}\n")
                self.result_text.insert(tk.END, f"AUC-ROC:   {auc_roc:.4f}\n")
                self.result_text.insert(tk.END, f"\nConfusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")
            else:
                self.result_text.insert(tk.END, "⚠️ No valid predictions\n")
                self.last_evaluation_metrics = None
            
            self.result_text.update()
            
            # 6. 결과 요약
            total = len(self.current_results)
            anomalies = sum(1 for r in self.current_results if r['is_anomaly'])
            normals = total - anomalies
            
            self.result_text.insert(tk.END, f"\n{'='*70}\n")
            self.result_text.insert(tk.END, f"✅ DETECTION COMPLETE\n")
            self.result_text.insert(tk.END, f"{'='*70}\n")
            self.result_text.insert(tk.END, f"Total: {total} | 🔴 Anomaly: {anomalies} | 🟢 Normal: {normals}\n")
            
            if self.last_evaluation_metrics:
                self.result_text.insert(tk.END, f"Performance: F1={self.last_evaluation_metrics['f1_score']:.4f}, Recall={self.last_evaluation_metrics['recall']:.4f}\n")
            
            self.result_text.insert(tk.END, f"{'='*70}\n")
            
            # 7. 자동 저장 (기존 코드 유지 - auto_generate_full_report 호출)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.auto_generate_full_report(timestamp)
            
            messagebox.showinfo(
                "Detection Complete",
                f"Processed: {total} files\n"
                f"Anomalies: {anomalies}\n"
                f"Normal: {normals}\n\n"
                f"F1-Score: {self.last_evaluation_metrics.get('f1_score', 0):.4f}\n"
                f"Recall: {self.last_evaluation_metrics.get('recall', 0):.4f}"
            )
            
        except Exception as e:
            self.result_text.insert(tk.END, f"\n❌ ERROR: {str(e)}\n")
            import traceback
            self.result_text.insert(tk.END, f"\n{traceback.format_exc()}\n")
            messagebox.showerror("Error", f"Detection failed:\n{str(e)}")
    
    def auto_export_results_csv(self, timestamp):
        """
        자동으로 CSV 저장 (reports 폴더에)
        
        Args:
            timestamp: 그래프 폴더와 동일한 timestamp (YYMMDD_HHMM)
        """
        if not self.current_results:
            self.log("⚠️ No results to export")
            return False
        
        if not self.training_data_dir:
            self.log("❌ Training data folder not set")
            return False
        
        try:
            # reports 폴더 생성
            report_dir = self.training_data_dir / "LSTM_IF" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일명 생성
            file_path = report_dir / f"lstmif_results_{timestamp}.csv"
            
            self.result_text.insert(tk.END, f"📄 Saving CSV: {file_path.name}...\n")
            self.result_text.update()
            
            export_data = []
            is_ensemble = any(r.get('detection_method') == 'ensemble' for r in self.current_results)
            
            for result in self.current_results:
                row = {
                    'file': result.get('file', 'N/A'),
                    'detection_method': result.get('detection_method', 'unknown'),
                    'is_anomaly': result.get('is_anomaly', False),
                    'vacuum_range': result.get('vacuum_range', 'N/A'),
                    'vacuum_value': result.get('vacuum_value', 0),
                    'confidence': result.get('confidence', 0)
                }
                
                if result.get('detection_method') == 'ensemble':
                    row.update({
                        'ensemble_score': result.get('ensemble_score', 0),
                        'ensemble_strategy': result.get('ensemble_strategy', 'N/A'),
                        'lstmae_normalized_score': result.get('lstmae_normalized_score', 0),
                        'if_normalized_score': result.get('if_normalized_score', 0),
                        'lstmae_weight': result.get('lstmae_weight', 0),
                        'if_weight': result.get('if_weight', 0)
                    })
                    
                    if 'lstmae_result' in result:
                        lstmae = result['lstmae_result']
                        row.update({
                            'lstmae_anomaly_score': lstmae.get('anomaly_score', 0),
                            'lstmae_threshold': lstmae.get('threshold', 0),
                            'lstmae_adjusted_threshold': lstmae.get('adjusted_threshold', 0),
                            'mse': lstmae.get('mse', 0),
                            'weighted_error': lstmae.get('weighted_error', 0),
                            'peak_time': lstmae.get('peak_time', 0)
                        })
                        
                        if 'anomaly_sensors' in lstmae:
                            row['anomaly_sensors'] = ', '.join(lstmae.get('anomaly_sensors', []))
                    
                    if 'if_result' in result:
                        if_res = result['if_result']
                        row.update({
                            'if_anomaly_probability': if_res.get('anomaly_probability', 0),
                            'if_raw_score': if_res.get('raw_score', 0)
                        })
                
                else:  # LSTM-AE only
                    row.update({
                        'anomaly_score': result.get('anomaly_score', 0),
                        'threshold': result.get('threshold', 0),
                        'adjusted_threshold': result.get('adjusted_threshold', 0),
                        'mse': result.get('mse', 0),
                        'weighted_error': result.get('weighted_error', 0),
                        'peak_time': result.get('peak_time', 0)
                    })
                    
                    if 'anomaly_sensors' in result:
                        row['anomaly_sensors'] = ', '.join(result.get('anomaly_sensors', []))
                
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            
            # 컬럼 순서 정리
            base_cols = ['file', 'detection_method', 'is_anomaly', 'vacuum_range', 'vacuum_value', 'confidence']
            other_cols = [col for col in df.columns if col not in base_cols]
            df = df[base_cols + other_cols]
            
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            self.result_text.insert(tk.END, f"✅ CSV saved: {file_path.name}\n")
            self.log(f"CSV exported: {file_path}")
            return True
            
        except Exception as e:
            self.result_text.insert(tk.END, f"❌ CSV save failed: {str(e)}\n")
            self.log(f"CSV export failed: {str(e)}")
            return False
    
    def auto_generate_full_report(self, timestamp):
        """
        Detection 결과를 CSV + JSON + HTML로 자동 저장
        ✅ Model 정보 + Evaluation 메트릭 + Results 전체 포함
        """
        try:
            # ✅ 저장 경로 통일
            results_dir = Path("D:/Results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            base_filename = f"detection_{timestamp}"
            
            # 1. ✅ CSV 저장
            csv_path = results_dir / f"{base_filename}.csv"
            df_results = pd.DataFrame(self.current_results)
            df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            self.result_text.insert(tk.END, f"\n✅ CSV saved: {csv_path.name}\n")
            
            # 2. ✅ JSON 저장 (전체 메타데이터)
            json_path = results_dir / f"{base_filename}_metadata.json"
            
            metadata = {
                'detection_info': {
                    'timestamp': timestamp,
                    'detection_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'total_files': len(self.current_results),
                    'anomaly_count': sum(1 for r in self.current_results if r['is_anomaly']),
                    'normal_count': sum(1 for r in self.current_results if not r['is_anomaly']),
                    'detection_method': self.detection_method.get()
                },
                'model_info': {},
                'evaluation_metrics': {},
                'results': self.current_results,
                'vacuum_range_statistics': {}
            }
            
            # ✅ Model 정보 추가
            if self.model:
                metadata['model_info'] = {
                    'model_id': self.model.model_id,
                    'created_at': self.model.created_at,
                    'sequence_length': self.model.sequence_length,
                    'n_features': self.model.n_features,
                    'encoder_units': self.model.encoder_units,
                    'latent_dim': self.model.latent_dim,
                    'decoder_units': self.model.decoder_units,
                    'use_bidirectional': self.model.use_bidirectional,
                    'dropout': self.model.dropout,
                    'recurrent_dropout': self.model.recurrent_dropout,
                    'learning_rate': self.model.lr,
                    'l2_reg': self.model.l2_reg,
                    'clip_norm': self.model.clip_norm,
                    'optimizer': self.model.optimizer,
                    'latent_activation': self.model.latent_activation,
                    'use_batch_norm': self.model.use_batch_norm,
                    'base_threshold': float(self.model.threshold) if self.model.threshold else None,
                    'training_time': float(self.model.training_time),
                    'best_epoch': int(self.model.best_epoch),
                    'train_mse_mean': float(self.model.train_stats.get('mean', 0)),
                    'train_mse_std': float(self.model.train_stats.get('std', 0)),
                    'train_mse_median': float(self.model.train_stats.get('median', 0)),
                    'train_mse_min': float(self.model.train_stats.get('min', 0)),
                    'train_mse_max': float(self.model.train_stats.get('max', 0)),
                    'val_mse_mean': float(self.model.val_stats.get('mean', 0)),
                    'val_mse_std': float(self.model.val_stats.get('std', 0)),
                    'val_mse_median': float(self.model.val_stats.get('median', 0)),
                    'val_mse_min': float(self.model.val_stats.get('min', 0)),
                    'val_mse_max': float(self.model.val_stats.get('max', 0)),
                    'vacuum_range_thresholds': {
                        range_name: float(threshold)
                        for range_name, threshold in self.model.vacuum_range_thresholds.items()
                    }
                }
                
                if hasattr(self.model, 'if_model') and self.model.if_model is not None:
                    metadata['model_info']['isolation_forest'] = {
                        'n_estimators': self.model.if_model.n_estimators,
                        'contamination': self.model.if_model.contamination,
                        'max_samples': self.model.if_model.max_samples
                    }
            
            # ✅ Evaluation 메트릭 추가
            if hasattr(self, 'last_evaluation_metrics') and self.last_evaluation_metrics:
                metadata['evaluation_metrics'] = self.last_evaluation_metrics
            
            # ✅ Vacuum Range별 통계
            range_stats = {}
            for result in self.current_results:
                vr = result.get('vacuum_range', 'UNKNOWN')
                if vr not in range_stats:
                    range_stats[vr] = {
                        'total': 0,
                        'anomaly_count': 0,
                        'normal_count': 0,
                        'anomaly_scores': []
                    }
                
                range_stats[vr]['total'] += 1
                if result['is_anomaly']:
                    range_stats[vr]['anomaly_count'] += 1
                else:
                    range_stats[vr]['normal_count'] += 1
                range_stats[vr]['anomaly_scores'].append(result.get('anomaly_score', 0))
            
            for vr, stats in range_stats.items():
                scores = stats.pop('anomaly_scores')
                stats['anomaly_rate'] = (stats['anomaly_count'] / stats['total'] * 100) if stats['total'] > 0 else 0
                stats['avg_anomaly_score'] = float(np.mean(scores)) if scores else 0
                stats['max_anomaly_score'] = float(np.max(scores)) if scores else 0
                stats['min_anomaly_score'] = float(np.min(scores)) if scores else 0
            
            metadata['vacuum_range_statistics'] = range_stats
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.result_text.insert(tk.END, f"✅ JSON saved: {json_path.name}\n")
            
            # 3. ✅ HTML 리포트 생성
            html_path = results_dir / f"{base_filename}_report.html"
            self.generate_full_report_to_file(html_path)
            
            self.result_text.insert(tk.END, f"✅ HTML saved: {html_path.name}\n")
            self.result_text.insert(tk.END, f"\n📁 All files saved to: {results_dir}\n")
            self.result_text.update()
            
            # 브라우저에서 HTML 열기
            try:
                import webbrowser
                webbrowser.open(str(html_path))
            except:
                pass
            
        except Exception as e:
            self.result_text.insert(tk.END, f"\n❌ Error saving results: {str(e)}\n")
            import traceback
            self.result_text.insert(tk.END, f"\n{traceback.format_exc()}\n")


# ========================================================================
# 수정 4: generate_full_report() 함수를 generate_full_report_to_file()로 수정
# 위치: Line 6298 함수 이름 변경 및 내용 전체 교체
# ========================================================================

    def generate_full_report_to_file(self, report_path):
        """
        포괄적인 HTML 리포트 생성
        ✅ Model 정보 + Evaluation 메트릭 + Results 전체 포함
        """
        
        is_ensemble = self.detection_method.get() == "Ensemble"
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Cryo Pump Anomaly Detection Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }
        h3 {
            color: #5a6c7d;
            margin: 25px 0 15px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:nth-child(even) { background-color: #f8f9fa; }
        tr:hover { background-color: #e9ecef; }
        .anomaly { color: #e74c3c; font-weight: bold; }
        .normal { color: #27ae60; font-weight: bold; }
        .info-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #3498db;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
            color: white;
        }
        .metric-card .value {
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }
        .summary-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin: 20px 0;
        }
        .summary-stats {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }
        .summary-stat .number {
            font-size: 48px;
            font-weight: bold;
        }
        .cm-table {
            width: 60%;
            margin: 20px auto;
        }
        .cm-table td, .cm-table th {
            text-align: center;
            padding: 15px;
            font-size: 16px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Cryo Pump Anomaly Detection Report</h1>
        <div class="info-box">
            <p><strong>📅 Generated:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p><strong>🔧 Detection Method:</strong> """ + self.detection_method.get() + """</p>
        </div>
"""
        
        # Summary
        total = len(self.current_results)
        anomalies = sum(1 for r in self.current_results if r['is_anomaly'])
        normals = total - anomalies
        anomaly_rate = (anomalies / total * 100) if total > 0 else 0
        
        html_content += f"""
        <div class="summary-box">
            <h2 style="color: white; border: none; margin-bottom: 20px;">📊 Detection Summary</h2>
            <div class="summary-stats">
                <div class="summary-stat">
                    <div class="number">{total}</div>
                    <div>Total Files</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{anomalies}</div>
                    <div>🔴 Anomalies</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{normals}</div>
                    <div>🟢 Normal</div>
                </div>
                <div class="summary-stat">
                    <div class="number">{anomaly_rate:.1f}%</div>
                    <div>Anomaly Rate</div>
                </div>
            </div>
        </div>
"""
        
        # Model Info
        html_content += "<h2>📋 Model Information</h2><table>"
        
        if self.model:
            html_content += f"""
            <tr><th style="width: 30%;">Property</th><th>Value</th></tr>
            <tr><td><strong>Model ID</strong></td><td>{self.model.model_id}</td></tr>
            <tr><td><strong>Created</strong></td><td>{self.model.created_at}</td></tr>
            <tr><td><strong>Architecture</strong></td><td>Encoder: {self.model.encoder_units} → Latent: {self.model.latent_dim} → Decoder: {self.model.decoder_units}</td></tr>
            <tr><td><strong>Bidirectional</strong></td><td>{'✅ Yes' if self.model.use_bidirectional else '❌ No'}</td></tr>
            <tr><td><strong>Batch Norm</strong></td><td>{'✅ Yes' if self.model.use_batch_norm else '❌ No'}</td></tr>
            <tr><td><strong>Dropout</strong></td><td>{self.model.dropout} / {self.model.recurrent_dropout}</td></tr>
            <tr><td><strong>Learning Rate</strong></td><td>{self.model.lr}</td></tr>
            <tr><td><strong>Optimizer</strong></td><td>{self.model.optimizer.upper()}</td></tr>
            <tr><td><strong>Base Threshold</strong></td><td>{self.model.threshold:.6f}</td></tr>
            <tr><td><strong>Training Time</strong></td><td>{self.model.training_time:.1f}s</td></tr>
            <tr><td><strong>Best Epoch</strong></td><td>{self.model.best_epoch}</td></tr>
            <tr><td colspan="2" style="background: #f1f3f5; font-weight: bold;">📈 Training Stats</td></tr>
            <tr><td><strong>Train MSE</strong></td><td>{self.model.train_stats.get('mean', 0):.6f} ± {self.model.train_stats.get('std', 0):.6f}</td></tr>
            <tr><td><strong>Val MSE</strong></td><td>{self.model.val_stats.get('mean', 0):.6f} ± {self.model.val_stats.get('std', 0):.6f}</td></tr>
"""
            
            if self.model.vacuum_range_thresholds:
                html_content += "<tr><td colspan='2' style='background: #f1f3f5; font-weight: bold;'>🎯 Vacuum Thresholds</td></tr>"
                for rng, thr in sorted(self.model.vacuum_range_thresholds.items()):
                    html_content += f"<tr><td><strong>  {rng}</strong></td><td>{thr:.6f}</td></tr>"
        
        html_content += "</table>"
        
        # Evaluation Metrics
        if hasattr(self, 'last_evaluation_metrics') and self.last_evaluation_metrics:
            metrics = self.last_evaluation_metrics
            
            html_content += """
        <h2>📊 Evaluation Metrics</h2>
        <div class="metric-grid">
"""
            
            html_content += f"""
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="value">{metrics['accuracy']:.3f}</div>
            </div>
            <div class="metric-card">
                <h3>Precision</h3>
                <div class="value">{metrics['precision']:.3f}</div>
            </div>
            <div class="metric-card">
                <h3>Recall</h3>
                <div class="value">{metrics['recall']:.3f}</div>
            </div>
            <div class="metric-card">
                <h3>F1-Score</h3>
                <div class="value">{metrics['f1_score']:.3f}</div>
            </div>
            <div class="metric-card">
                <h3>FPR</h3>
                <div class="value">{metrics['false_positive_rate']:.3f}</div>
            </div>
            <div class="metric-card">
                <h3>AUC-ROC</h3>
                <div class="value">{metrics.get('auc_roc', 0):.3f}</div>
            </div>
"""
            
            html_content += f"""
        </div>
        <h3>Confusion Matrix</h3>
        <table class="cm-table">
            <tr><th></th><th>Pred Normal</th><th>Pred Anomaly</th></tr>
            <tr><th>Actual Normal</th>
                <td class="normal" style="font-size: 24px;">{metrics['confusion_matrix']['true_negative']}</td>
                <td class="anomaly" style="font-size: 24px;">{metrics['confusion_matrix']['false_positive']}</td>
            </tr>
            <tr><th>Actual Anomaly</th>
                <td class="anomaly" style="font-size: 24px;">{metrics['confusion_matrix']['false_negative']}</td>
                <td class="normal" style="font-size: 24px;">{metrics['confusion_matrix']['true_positive']}</td>
            </tr>
        </table>
"""
        
        # Detection Results
        html_content += "<h2>🔍 Detection Results</h2><table>"
        
        if is_ensemble:
            html_content += """
            <tr><th>File</th><th>Vacuum Range</th><th>Status</th><th>Ensemble</th><th>LSTM-AE</th><th>IF</th><th>Conf</th></tr>
"""
        else:
            html_content += """
            <tr><th>File</th><th>Vacuum Range</th><th>Status</th><th>Score</th><th>Threshold</th><th>Confidence</th></tr>
"""
        
        for result in self.current_results:
            status_class = 'anomaly' if result['is_anomaly'] else 'normal'
            status_text = '🔴 ANOMALY' if result['is_anomaly'] else '🟢 NORMAL'
            
            if is_ensemble:
                html_content += f"""
            <tr>
                <td>{result.get('file', 'N/A')}</td>
                <td>{result.get('vacuum_range', 'N/A')}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{result.get('ensemble_score', 0):.4f}</td>
                <td>{result.get('lstmae_normalized_score', 0):.4f}</td>
                <td>{result.get('if_normalized_score', 0):.4f}</td>
                <td>{result.get('confidence', 0)*100:.1f}%</td>
            </tr>
"""
            else:
                html_content += f"""
            <tr>
                <td>{result.get('file', 'N/A')}</td>
                <td>{result.get('vacuum_range', 'N/A')}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{result.get('anomaly_score', 0):.6f}</td>
                <td>{result.get('adjusted_threshold', 0):.6f}</td>
                <td>{result.get('confidence', 0)*100:.1f}%</td>
            </tr>
"""
        
        html_content += "</table>"
        
        # Vacuum Range Stats
        html_content += """
        <h2>📈 Vacuum Range Statistics</h2>
        <table>
            <tr><th>Range</th><th>Total</th><th>Anomalies</th><th>Normal</th><th>Rate</th></tr>
"""
        
        range_stats = {}
        for result in self.current_results:
            vr = result.get('vacuum_range', 'UNKNOWN')
            if vr not in range_stats:
                range_stats[vr] = {'total': 0, 'anomaly': 0, 'normal': 0}
            range_stats[vr]['total'] += 1
            if result['is_anomaly']:
                range_stats[vr]['anomaly'] += 1
            else:
                range_stats[vr]['normal'] += 1
        
        for vr, stats in sorted(range_stats.items()):
            rate = stats['anomaly'] / stats['total'] * 100 if stats['total'] > 0 else 0
            html_content += f"""
            <tr>
                <td><strong>{vr}</strong></td>
                <td>{stats['total']}</td>
                <td class="anomaly">{stats['anomaly']}</td>
                <td class="normal">{stats['normal']}</td>
                <td>{rate:.1f}%</td>
            </tr>
"""
        
        html_content += """
        </table>
        <div class="footer">
            <p>Cryo Pump AI Anomaly Detection System</p>
            <p>EV Tech Team | Samsung Electronics</p>
            <p>Created by SeJun Kang</p>
        </div>
    </div>
</body>
</html>
"""
        
        # 파일 저장
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        except Exception as e:
            print(f"Error saving HTML: {str(e)}")

    
    def _save_lstmae_ai_pro_simple(self, original_data, result, save_path):
        """
        AI Pro 심사용: 한눈에 이해되는 LSTM-AE 결과 시각화 (5-panel)
        """
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
            
            # 제목
            status = "🔴 ANOMALY DETECTED" if result['is_anomaly'] else "🟢 NORMAL"
            fig.suptitle(f'LSTM-AE Anomaly Detection: {result["file"]}\n{status}', 
                         fontsize=16, fontweight='bold')
            
            # 데이터 준비
            time_errors = np.array(result.get('time_errors', []))
            if len(time_errors) == 0:
                # time_errors가 없으면 직접 계산
                processed = self.model.preprocess_data(original_data)
                downsampled = self.model.downsample(processed)
                downsampled_reshaped = downsampled.reshape(1, self.model.sequence_length, self.model.n_features)
                prediction = self.model.autoencoder.predict(downsampled_reshaped, verbose=0)
                errors = np.square(downsampled_reshaped - prediction)
                time_errors = np.mean(errors[0], axis=1)
            
            threshold = result['adjusted_threshold']
            sensor_contributions = result.get('sensor_contributions', [0]*7)
            sensor_names = ['Vacuum', 'Cryo1', 'Cryo2', 'Cryo3', 'Cryo4', 'Comp1', 'Comp2']
            
            # ==================== [1] Reconstruction Error (메인) ====================
            ax1 = fig.add_subplot(gs[0, :])
            
            time_axis = np.arange(len(time_errors)) * 0.5  # 30초 간격 → 분 단위
            
            ax1.plot(time_axis, time_errors, linewidth=1.5, color='#2196F3', 
                     label='Reconstruction Error', alpha=0.8)
            
            # Threshold
            ax1.axhline(threshold, color='red', linestyle='--', linewidth=2,
                        label=f'Threshold: {threshold:.6f}')
            
            # 이상 구간 강조
            anomaly_mask = time_errors > threshold
            if np.any(anomaly_mask):
                ax1.fill_between(time_axis, 0, time_errors, 
                                 where=anomaly_mask, alpha=0.3, color='red',
                                 label='Anomaly Region')
            
            ax1.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
            ax1.set_title('Temporal Reconstruction Error (Downsampled: 7000→700 points)', 
                          fontsize=13, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 350)
            
            # ==================== [2] 10구간 집계 ====================
            ax2 = fig.add_subplot(gs[1, 0])
            
            # 70포인트씩 10구간으로 집계
            segment_errors = []
            for i in range(10):
                start = i * 70
                end = min(start + 70, len(time_errors))
                if end > start:
                    segment_error = np.mean(time_errors[start:end])
                    segment_errors.append(segment_error)
                else:
                    segment_errors.append(0)
            
            colors = ['red' if e > threshold else 'green' for e in segment_errors]
            
            bars = ax2.bar(range(10), segment_errors, color=colors, alpha=0.6, edgecolor='black')
            ax2.axhline(threshold, color='darkred', linestyle='--', linewidth=2)
            ax2.set_xticks(range(10))
            ax2.set_xticklabels([f'S{i+1}' for i in range(10)])
            ax2.set_xlabel('Time Segment (35min each)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Avg Error', fontsize=11, fontweight='bold')
            ax2.set_title('Error by Time Segment', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # ==================== [3] 센서별 기여도 ====================
            ax3 = fig.add_subplot(gs[1, 1])
            
            colors_sensor = ['#D32F2F' if c > 0.2 else '#388E3C' for c in sensor_contributions]
            bars_sensor = ax3.barh(sensor_names, sensor_contributions, color=colors_sensor, alpha=0.7)
            ax3.axvline(0.2, color='red', linestyle='--', linewidth=2, label='High Contribution (>20%)')
            ax3.set_xlabel('Contribution to Anomaly', fontsize=11, fontweight='bold')
            ax3.set_title('Sensor Contribution Analysis', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, axis='x')
            
            # ==================== [4] 원본 Vacuum Pressure ====================
            ax4 = fig.add_subplot(gs[2, 0])
            
            time_orig = np.arange(len(original_data)) * 3 / 60  # 분 단위
            ax4.plot(time_orig, original_data[:, 0], linewidth=0.5, color='purple', alpha=0.7)
            
            # 이상 구간 배경색
            for i, err in enumerate(segment_errors):
                if err > threshold:
                    ax4.axvspan(i*35, (i+1)*35, alpha=0.2, color='red')
            
            ax4.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Vacuum Pressure (Torr)', fontsize=11, fontweight='bold')
            ax4.set_title('Original Vacuum Sensor Data (7000 points)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, 350)
            ax4.set_yscale('log')
            
            # ==================== [5] 통계 정보 박스 ====================
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.axis('off')
            
            anomalous_segments = sum(1 for e in segment_errors if e > threshold)
            peak_time_min = result.get('peak_time', 0) * 0.5
            
            stats_text = f"""
📊 DETECTION SUMMARY

Overall Status: {'ANOMALY' if result['is_anomaly'] else 'NORMAL'}
Anomaly Score: {result['anomaly_score']:.6f}
Threshold: {result['adjusted_threshold']:.6f}
Confidence: {result['confidence']*100:.1f}%

Vacuum Range: {result['vacuum_range']}
Vacuum Value: {result['vacuum_value']:.2e} Torr

⏱️ TIME ANALYSIS
Peak Error Time: {peak_time_min:.1f} min
Anomalous Segments: {anomalous_segments}/10

🔍 SENSOR ANALYSIS
Primary Anomaly Sensors:
{', '.join(result.get('anomaly_sensors', ['None']))}

📈 ERROR METRICS
Mean Squared Error: {result['mse']:.6f}
Weighted Error: {result['weighted_error']:.6f}
            """
            
            ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
                     fontsize=10, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            if 'fig' in locals():
                plt.close(fig)
            raise e
            
    def save_detection_plot(self, original_data, result, use_ensemble, save_path):
            """
            이상탐지 결과를 시계열 기반으로 시각화하는 개선된 그래프 생성
            
            Args:
                original_data: 원본 센서 데이터 (7000, 7)
                result: detection 결과 딕셔너리
                use_ensemble: Ensemble 사용 여부
                save_path: 저장 경로 (Path 객체)
            """
            try:
                # ========== AI Pro 심사용: LSTM-AE 단독 간결 버전 ==========
                if not use_ensemble:
                    self._save_lstmae_ai_pro_simple(original_data, result, save_path)
                    return
                
                # ========== 폰트 설정 (한글 깨짐 방지) ==========
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.rcParams['axes.unicode_minus'] = False
                
                # ========== 데이터 준비 ==========
                processed = self.model.preprocess_data(original_data)
                downsampled = self.model.downsample(processed)
                downsampled_reshaped = downsampled.reshape(1, self.model.sequence_length, self.model.n_features)
                
                prediction = self.model.autoencoder.predict(downsampled_reshaped, verbose=0)
                
                # Error 계산
                errors = np.square(downsampled_reshaped - prediction)
                time_mse = np.mean(errors, axis=2)[0]  # (700,) - 시간축 전체 MSE
                sensor_errors = np.mean(errors, axis=1)[0]  # (7,) - 센서별 평균 에러
                
                # Time axis
                time_axis = np.arange(len(time_mse))
                
                # 센서 이름
                sensor_names = ['Vacuum', 'Pump1 Temp', 'Pump2 Temp', 'Pump3 Temp',
                               'Pump4 Temp', 'Comp1 Press', 'Comp2 Press']
                
                # ========== Top-3 센서 찾기 ==========
                top3_indices = np.argsort(sensor_errors)[-3:][::-1]
                top3_sensors = [(sensor_names[i], sensor_errors[i], i) for i in top3_indices]
                
                # ========== Figure 생성 (4x1 레이아웃) ==========
                fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
                fig.subplots_adjust(hspace=0.35, left=0.08, right=0.95, top=0.94, bottom=0.06)
                
                # ========== Title ==========
                is_anomaly = result['is_anomaly']
                title_color = '#C62828' if is_anomaly else '#2E7D32'
                status_text = 'ANOMALY DETECTED' if is_anomaly else 'NORMAL'
                status_icon = '🔴' if is_anomaly else '🟢'
                
                # 메인 타이틀
                confidence = result.get('confidence', 0) * 100
                vacuum_range = result.get('vacuum_range', 'N/A')
                
                fig.suptitle(
                    f'{status_icon} {status_text} | {result["file"]} | '
                    f'Range: {vacuum_range} | Confidence: {confidence:.1f}%',
                    fontsize=18, fontweight='bold', color=title_color, y=0.98
                )
                
                # ========== [0] LSTM-AE Reconstruction Error (Time-series) ==========
                ax0 = axes[0]
                
                # Threshold 추출
                if use_ensemble:
                    lstmae_threshold = result['lstmae_result']['adjusted_threshold']
                else:
                    lstmae_threshold = result['adjusted_threshold']
                
                # MSE 라인 플롯
                ax0.plot(time_axis, time_mse, color='#1976D2', linewidth=2, label='Reconstruction Error', alpha=0.9)
                
                # Threshold 라인
                ax0.axhline(lstmae_threshold, color='red', linestyle='--', linewidth=2.5, 
                           label=f'Threshold: {lstmae_threshold:.6f}', alpha=0.8, zorder=3)
                
                # 배경 색상 구역 (threshold 기준)
                ax0.axhspan(0, lstmae_threshold, alpha=0.08, color='green', zorder=1)
                y_max = max(time_mse.max(), lstmae_threshold * 1.5)
                ax0.axhspan(lstmae_threshold, y_max, alpha=0.08, color='red', zorder=1)
                
                # Peak 표시
                peak_time = result.get('peak_time') or (result.get('lstmae_result', {}).get('peak_time'))
                if peak_time is not None and 0 <= peak_time < len(time_mse):
                    ax0.axvline(peak_time, color='orange', linestyle=':', linewidth=2, 
                               label=f'Peak at t={peak_time}', alpha=0.7, zorder=2)
                    ax0.scatter([peak_time], [time_mse[peak_time]], color='red', s=300, 
                               zorder=5, edgecolors='black', linewidth=2, marker='*')
                
                # Anomaly 구간 강조 (threshold 초과 구간)
                anomaly_mask = time_mse > lstmae_threshold
                if np.any(anomaly_mask):
                    ax0.fill_between(time_axis, 0, time_mse, where=anomaly_mask, 
                                    color='red', alpha=0.15, label='Anomaly Zone', zorder=2)
                
                ax0.set_ylabel('Mean Squared Error', fontsize=11, fontweight='bold')
                ax0.set_title('[1] LSTM-AE Reconstruction Error over Time', 
                             fontsize=13, fontweight='bold', pad=12, loc='left')
                ax0.legend(loc='upper right', fontsize=9, framealpha=0.95)
                ax0.grid(True, alpha=0.25, linestyle='--')
                ax0.set_xlim(0, len(time_axis))
                
                # Score 정보 추가
                if use_ensemble:
                    score_text = f"LSTM-AE Score: {result['lstmae_normalized_score']:.4f}"
                else:
                    score_text = f"Anomaly Score: {result['anomaly_score']:.6f}"
                
                ax0.text(0.02, 0.97, score_text, transform=ax0.transAxes, 
                        fontsize=10, va='top', ha='left', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8, edgecolor='black'))
                
                # ========== [1] Isolation Forest Score Band ==========
                ax1 = axes[1]
                
                if use_ensemble:
                    if_score = result['if_normalized_score']
                    if_threshold = 1.0
                    
                    # IF는 시간 독립적이므로 horizontal band로 표시
                    ax1.axhline(if_score, color='#F57C00', linewidth=3, 
                               label=f'IF Score: {if_score:.4f}', alpha=0.9, zorder=3)
                    ax1.axhline(if_threshold, color='red', linestyle='--', linewidth=2.5, 
                               label=f'IF Threshold: {if_threshold:.2f}', alpha=0.8, zorder=3)
                    
                    # Score가 threshold 초과하면 빨간 배경
                    if if_score > if_threshold:
                        ax1.axhspan(if_threshold, if_score, alpha=0.15, color='red', zorder=1)
                        ax1.axhspan(0, if_threshold, alpha=0.08, color='green', zorder=1)
                    else:
                        ax1.axhspan(0, if_threshold, alpha=0.08, color='green', zorder=1)
                    
                    # 전체 시간축에 걸쳐 band 표시
                    ax1.fill_between(time_axis, if_score - 0.02, if_score + 0.02, 
                                    alpha=0.3, color='#F57C00', zorder=2)
                    
                    ax1.set_ylabel('IF Anomaly Score', fontsize=11, fontweight='bold')
                    ax1.set_title('[2] Isolation Forest Anomaly Score (Time-independent)', 
                                 fontsize=13, fontweight='bold', pad=12, loc='left')
                    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
                    ax1.set_ylim(0, max(1.5, if_score + 0.3))
                    
                else:
                    # Ensemble이 아닐 때는 "LSTM-AE Only" 메시지 표시
                    ax1.text(0.5, 0.5, 'Isolation Forest Not Used\n(LSTM-AE Detection Only)', 
                            transform=ax1.transAxes, fontsize=13, ha='center', va='center',
                            fontweight='bold', color='gray',
                            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', 
                                     alpha=0.3, edgecolor='gray', linewidth=2))
                    ax1.set_ylabel('N/A', fontsize=11, fontweight='bold')
                    ax1.set_title('[2] Isolation Forest Score (Not Available)', 
                                 fontsize=13, fontweight='bold', pad=12, loc='left')
                    ax1.set_ylim(0, 1)
                
                ax1.grid(True, alpha=0.25, linestyle='--')
                ax1.set_xlim(0, len(time_axis))
                
                # ========== [2] Ensemble Final Score (메인 플롯) ==========
                ax2 = axes[2]
                
                if use_ensemble:
                    ensemble_score = result['ensemble_score']
                    ensemble_threshold = 1.0
                    
                    # Ensemble score를 시계열로 표현
                    lstmae_normalized = time_mse / lstmae_threshold if lstmae_threshold > 0 else time_mse
                    
                    # IF score는 상수이므로 가중 평균
                    lstmae_weight = result.get('lstmae_weight', 0.7)
                    if_weight = result.get('if_weight', 0.3)
                    
                    # Time-series ensemble score
                    ensemble_time_score = lstmae_weight * lstmae_normalized + if_weight * if_score
                    
                    # Color mapping
                    colors = []
                    for score in ensemble_time_score:
                        if score < 0.7:
                            colors.append('#2E7D32')  # Green
                        elif score < 1.0:
                            colors.append('#FFA726')  # Orange
                        else:
                            colors.append('#C62828')  # Red
                    
                    # Line plot with color segments
                    for i in range(len(time_axis) - 1):
                        ax2.plot(time_axis[i:i+2], ensemble_time_score[i:i+2], 
                                color=colors[i], linewidth=2.5, alpha=0.9)
                    
                    # Threshold 라인
                    ax2.axhline(ensemble_threshold, color='red', linestyle='--', linewidth=3, 
                               label=f'Threshold: {ensemble_threshold:.2f}', alpha=0.9, zorder=3)
                    
                    # 배경 색상 구역
                    ax2.axhspan(0, 0.7, alpha=0.05, color='green', zorder=1, label='Safe Zone')
                    ax2.axhspan(0.7, 1.0, alpha=0.05, color='yellow', zorder=1, label='Warning Zone')
                    y_max2 = max(ensemble_time_score.max(), 1.5)
                    ax2.axhspan(1.0, y_max2, alpha=0.08, color='red', zorder=1, label='Anomaly Zone')
                    
                    # Final score marker (최대값 위치)
                    max_idx = np.argmax(ensemble_time_score)
                    ax2.scatter([max_idx], [ensemble_time_score[max_idx]], 
                               color='darkred', s=350, zorder=5, 
                               edgecolors='black', linewidth=2.5, marker='D',
                               label=f'Peak: {ensemble_time_score[max_idx]:.3f}')
                    
                    ax2.set_ylabel('Ensemble Score', fontsize=11, fontweight='bold')
                    ax2.set_title('[3] Final Ensemble Score (Weighted Combination)', 
                                 fontsize=13, fontweight='bold', pad=12, loc='left')
                    
                    # Ensemble 정보 텍스트
                    ensemble_info = (f"Final Score: {ensemble_score:.4f}\n"
                                   f"Strategy: {result.get('ensemble_strategy', 'N/A')}\n"
                                   f"Weights: LSTM={lstmae_weight:.2f}, IF={if_weight:.2f}")
                    
                else:
                    # LSTM-AE only인 경우
                    anomaly_score = result['anomaly_score']
                    threshold = result['adjusted_threshold']
                    
                    # Normalized score
                    normalized_score = time_mse / threshold if threshold > 0 else time_mse
                    
                    # Color mapping
                    colors = []
                    for score in normalized_score:
                        if score < 0.7:
                            colors.append('#2E7D32')
                        elif score < 1.0:
                            colors.append('#FFA726')
                        else:
                            colors.append('#C62828')
                    
                    # Line plot
                    for i in range(len(time_axis) - 1):
                        ax2.plot(time_axis[i:i+2], normalized_score[i:i+2], 
                                color=colors[i], linewidth=2.5, alpha=0.9)
                    
                    # Threshold 라인
                    ax2.axhline(1.0, color='red', linestyle='--', linewidth=3, 
                               label=f'Threshold: 1.0 (normalized)', alpha=0.9, zorder=3)
                    
                    # 배경 구역
                    ax2.axhspan(0, 0.7, alpha=0.05, color='green', zorder=1, label='Safe')
                    ax2.axhspan(0.7, 1.0, alpha=0.05, color='yellow', zorder=1, label='Warning')
                    y_max2 = max(normalized_score.max(), 1.5)
                    ax2.axhspan(1.0, y_max2, alpha=0.08, color='red', zorder=1, label='Anomaly')
                    
                    # Peak marker
                    max_idx = np.argmax(normalized_score)
                    ax2.scatter([max_idx], [normalized_score[max_idx]], 
                               color='darkred', s=350, zorder=5, 
                               edgecolors='black', linewidth=2.5, marker='D',
                               label=f'Peak: {normalized_score[max_idx]:.3f}')
                    
                    ax2.set_ylabel('Normalized Score', fontsize=11, fontweight='bold')
                    ax2.set_title('[3] LSTM-AE Detection Score (Normalized)', 
                                 fontsize=13, fontweight='bold', pad=12, loc='left')
                    
                    ensemble_info = (f"Score: {anomaly_score:.6f}\n"
                                   f"Threshold: {threshold:.6f}\n"
                                   f"Ratio: {anomaly_score/threshold:.2f}x")
                
                # Score 정보 박스
                ax2.text(0.02, 0.97, ensemble_info, transform=ax2.transAxes, 
                        fontsize=9, va='top', ha='left', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', 
                                 alpha=0.85, edgecolor='black', linewidth=1.5))
                
                ax2.legend(loc='upper right', fontsize=8, framealpha=0.95, ncol=2)
                ax2.grid(True, alpha=0.25, linestyle='--')
                ax2.set_xlim(0, len(time_axis))
                
                # ========== [3] Top-3 Critical Sensor Contribution ==========
                ax3 = axes[3]
                
                # Top-3 센서 바 차트
                sensor_labels = [name for name, _, _ in top3_sensors]
                sensor_values = [error for _, error, _ in top3_sensors]
                
                # 색상 맵핑
                max_error = max(sensor_values) if sensor_values else 1
                bar_colors = [plt.cm.RdYlGn_r(val / max_error) for val in sensor_values]
                
                bars = ax3.barh(sensor_labels, sensor_values, color=bar_colors, 
                               edgecolor='black', linewidth=2, height=0.6)
                
                # 값 표시
                for bar, val in zip(bars, sensor_values):
                    width = bar.get_width()
                    ax3.text(width * 1.02, bar.get_y() + bar.get_height()/2,
                            f'{val:.5f}', va='center', ha='left', 
                            fontsize=10, fontweight='bold')
                
                # 나머지 센서 합계
                other_sensors_sum = np.sum([sensor_errors[i] for i in range(7) if i not in top3_indices])
                other_count = 7 - len(top3_indices)
                
                if other_count > 0:
                    ax3.text(0.98, 0.05, f'Others ({other_count} sensors): {other_sensors_sum:.5f}',
                            transform=ax3.transAxes, fontsize=9, va='bottom', ha='right',
                            style='italic', color='gray',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', 
                                     alpha=0.5, edgecolor='none'))
                
                ax3.set_xlabel('Mean Error (MSE)', fontsize=11, fontweight='bold')
                ax3.set_title('[4] Top-3 Critical Sensor Contribution', 
                             fontsize=13, fontweight='bold', pad=12, loc='left')
                ax3.grid(True, alpha=0.25, axis='x', linestyle='--')
                
                # x축 공유 - 맨 아래만 라벨
                ax3.set_xlabel('Time Step (Downsampled to 700)', fontsize=11, fontweight='bold')
                
                # ========== 저장 ==========
                plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.3)
                plt.close(fig)
                
                # 폰트 설정 복원
                plt.rcParams.update(plt.rcParamsDefault)
                
            except Exception as e:
                # 에러 발생 시 figure 정리
                if 'fig' in locals():
                    plt.close(fig)
                # 폰트 설정 복원
                plt.rcParams.update(plt.rcParamsDefault)
                raise e
    def create_visualization(self):
        viz_type = self.viz_type_var.get()
        self.viz_figure.clear()
        
        try:
            if viz_type == "training_history":
                self.plot_training_history()
            elif viz_type == "error_distribution":
                self.plot_error_distribution()
            elif viz_type == "vacuum_range_distribution":
                self.plot_vacuum_range_distribution()
            elif viz_type == "sensor_contribution":
                self.plot_sensor_contribution()
            elif viz_type == "threshold_analysis":
                self.plot_threshold_analysis()
            elif viz_type == "ensemble_comparison":
                self.plot_ensemble_comparison()                
            
            self.viz_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")
    
    def plot_training_history(self):
        if self.model is None or not hasattr(self.model, 'history'):
            messagebox.showwarning("Warning", "No training history available.")
            return
        
        ax = self.viz_figure.add_subplot(111)
        epochs = range(1, len(self.model.history['loss']) + 1)
        
        ax.plot(epochs, self.model.history['loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.8)
        ax.plot(epochs, self.model.history['val_loss'], 'r-', linewidth=2, label='Val Loss', alpha=0.8)
        
        if self.model.best_epoch > 0:
            ax.axvline(self.model.best_epoch, color='green', linestyle='--', alpha=0.5, 
                      label=f'Best: {self.model.best_epoch}')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training History\nModel: {self.model.model_id}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.viz_figure.tight_layout()
    
    def plot_error_distribution(self):
        if not self.current_results:
            messagebox.showwarning("Warning", "No detection results available.")
            return
        
        ax = self.viz_figure.add_subplot(111)
        
        # Detection method 확인
        is_ensemble = self.current_results[0].get('detection_method') == 'ensemble' if self.current_results else False
        
        if is_ensemble:
            # Ensemble score 사용
            scores = [r['ensemble_score'] for r in self.current_results]
            xlabel = 'Ensemble Score'
            title = 'Ensemble Score Distribution'
        else:
            # LSTM-AE anomaly score 사용
            scores = [r.get('anomaly_score', 0) for r in self.current_results]
            xlabel = 'Anomaly Score'
            title = 'Error Distribution'
        
        is_anomaly = [r['is_anomaly'] for r in self.current_results]
        
        normal = [s for s, a in zip(scores, is_anomaly) if not a]
        anomaly = [s for s, a in zip(scores, is_anomaly) if a]
        
        if normal:
            ax.hist(normal, bins=30, alpha=0.7, color='green', edgecolor='black', label='Normal', linewidth=1.2)
        if anomaly:
            ax.hist(anomaly, bins=30, alpha=0.7, color='red', edgecolor='black', label='Anomaly', linewidth=1.2)
        
        # Threshold line (ensemble의 경우 1.0)
        if is_ensemble:
            ax.axvline(x=1.0, color='blue', linestyle='--', linewidth=2, label='Threshold (1.0)', alpha=0.7)
        
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        self.viz_figure.tight_layout()
    
    def plot_vacuum_range_distribution(self):
        try:  # ✅ 함수 시작 부분에 try 추가
            if not self.current_results:
                messagebox.showwarning("Warning", "No detection results available.")
                return
            
            gs = GridSpec(2, 2, figure=self.viz_figure)
            range_data = {}
            
            # Detection method 확인
            is_ensemble = self.current_results[0].get('detection_method') == 'ensemble' if self.current_results else False
            
            for result in self.current_results:
                rng = result['vacuum_range']
                if rng not in range_data:
                    range_data[rng] = {'normal': [], 'anomaly': []}
                
                # Score 추출 - detection method에 따라
                if is_ensemble:
                    score = result.get('ensemble_score', 0)
                else:
                    score = result.get('anomaly_score', 0)
                
                if result['is_anomaly']:
                    range_data[rng]['anomaly'].append(score)
                else:
                    range_data[rng]['normal'].append(score)
            
            for idx, (rng, data) in enumerate(sorted(range_data.items())):
                if idx >= 4:
                    break
                ax = self.viz_figure.add_subplot(gs[idx // 2, idx % 2])
                
                if data['normal']:
                    ax.hist(data['normal'], bins=20, alpha=0.7, color='green', edgecolor='black')
                if data['anomaly']:
                    ax.hist(data['anomaly'], bins=20, alpha=0.7, color='red', edgecolor='black')
                
                title = f'{rng}'
                if is_ensemble:
                    title += '\n(Ensemble Score)'
                ax.set_title(title, fontsize=10)
                ax.grid(True, alpha=0.3)
            
            main_title = 'Results by Vacuum Range'
            if is_ensemble:
                main_title += ' - Ensemble Detection'
            self.viz_figure.suptitle(main_title, fontsize=14, fontweight='bold')
            self.viz_figure.tight_layout()
        
        except Exception as e:  # ✅ 이제 try와 짝이 맞음
            ax = self.viz_figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error generating plot:\n{str(e)}', 
                    ha='center', va='center', fontsize=12, color='red')
            ax.set_title('Vacuum Range Distribution', fontsize=14, fontweight='bold')
            self.viz_figure.tight_layout()
            messagebox.showerror("Error", f"Failed to generate plot:\n{str(e)}")     
    
    def plot_sensor_contribution(self):
        if not self.current_results:
            return
        
        ax = self.viz_figure.add_subplot(111)
        
        sensor_names = ['Vacuum', 'Pump1_Temp', 'Pump2_Temp', 'Pump3_Temp',
                       'Pump4_Temp', 'Comp1_Press', 'Comp2_Press']
        
        total_contrib = np.zeros(7)
        count = 0
        
        # Detection method 확인
        is_ensemble = self.current_results[0].get('detection_method') == 'ensemble' if self.current_results else False
        
        for result in self.current_results:
            if not result['is_anomaly']:
                continue
            
            # Ensemble인 경우 lstmae_result에서 가져오기
            if is_ensemble and 'lstmae_result' in result:
                sensor_contrib = result['lstmae_result'].get('sensor_contributions')
            else:
                sensor_contrib = result.get('sensor_contributions')
            
            if sensor_contrib:
                total_contrib += np.array(sensor_contrib)
                count += 1
        
        if count > 0:
            avg_contrib = total_contrib / count
            colors = plt.cm.RdYlGn_r(avg_contrib / max(avg_contrib))
            bars = ax.bar(sensor_names, avg_contrib, color=colors, edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, avg_contrib):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_ylabel('Average Contribution', fontsize=12, fontweight='bold')
            title = 'Sensor Contribution to Anomalies'
            if is_ensemble:
                title += ' (LSTM-AE Component)'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No anomaly data with sensor contributions available', 
                    ha='center', va='center', fontsize=12)
            ax.set_title('Sensor Contribution to Anomalies', fontsize=14, fontweight='bold')
        
        self.viz_figure.tight_layout()
    
    def plot_threshold_analysis(self):
        if self.model is None:
            return
        
        ax = self.viz_figure.add_subplot(111)
        
        ranges = list(self.model.vacuum_range_thresholds.keys())
        adjustments = list(self.model.vacuum_range_thresholds.values())
        thresholds = [self.model.threshold * adj for adj in adjustments]
        
        x = np.arange(len(ranges))
        width = 0.35
        
        ax.bar(x - width/2, [self.model.threshold] * len(ranges), width, 
              label='Base Threshold', color='#1976D2', alpha=0.7)
        ax.bar(x + width/2, thresholds, width, 
              label='Adjusted Threshold', color='#C62828', alpha=0.7)
        
        for i, (adj, thr) in enumerate(zip(adjustments, thresholds)):
            ax.text(i, thr, f'{adj:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Vacuum Range', fontsize=12)
        ax.set_ylabel('Threshold Value', fontsize=12)
        ax.set_title('Threshold Analysis by Vacuum Range', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(ranges)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        self.viz_figure.tight_layout()
        
    def plot_ensemble_comparison(self):
        """Compare LSTM-AE vs IF vs Ensemble predictions"""
        if not self.current_results:
            messagebox.showwarning("Warning", "No detection results available.")
            return
        
        # Check if results have ensemble data
        has_ensemble = any('detection_method' in r and r['detection_method'] == 'ensemble' 
                           for r in self.current_results)
        
        if not has_ensemble:
            messagebox.showwarning("Warning", "No ensemble results available. Train with ensemble enabled.")
            return
        
        fig = self.viz_figure
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract data
        lstmae_scores = []
        if_scores = []
        ensemble_scores = []
        labels = []
        
        for r in self.current_results:
            if 'lstmae_result' in r:
                lstmae_scores.append(r['lstmae_normalized_score'])
                if_scores.append(r['if_normalized_score'])
                ensemble_scores.append(r['ensemble_score'])
                labels.append(1 if r['is_anomaly'] else 0)
        
        lstmae_scores = np.array(lstmae_scores)
        if_scores = np.array(if_scores)
        ensemble_scores = np.array(ensemble_scores)
        labels = np.array(labels)
        
        # Plot 1: Score Comparison Scatter
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ['red' if l == 1 else 'green' for l in labels]
        ax1.scatter(lstmae_scores, if_scores, c=colors, alpha=0.6, s=100, edgecolors='black')
        ax1.plot([0, 2], [0, 2], 'k--', alpha=0.3)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='IF Threshold')
        ax1.axvline(x=1.0, color='blue', linestyle='--', alpha=0.5, label='LSTM-AE Threshold')
        ax1.set_xlabel('LSTM-AE Normalized Score', fontweight='bold')
        ax1.set_ylabel('IF Normalized Score', fontweight='bold')
        ax1.set_title('Score Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Score Distributions
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(lstmae_scores))
        width = 0.25
        ax2.bar(x - width, lstmae_scores, width, label='LSTM-AE', alpha=0.7, color='#1976D2')
        ax2.bar(x, if_scores, width, label='IF', alpha=0.7, color='#F57C00')
        ax2.bar(x + width, ensemble_scores, width, label='Ensemble', alpha=0.7, color='#2E7D32')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax2.set_xlabel('Sample Index', fontweight='bold')
        ax2.set_ylabel('Normalized Score', fontweight='bold')
        ax2.set_title('All Scores by Sample', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Agreement Analysis
        ax3 = fig.add_subplot(gs[1, 0])
        agreement = np.abs(lstmae_scores - if_scores)
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
        ax3.hist(agreement, bins=bins, edgecolor='black', alpha=0.7, color='purple')
        ax3.set_xlabel('Score Difference |LSTM-AE - IF|', fontweight='bold')
        ax3.set_ylabel('Count', fontweight='bold')
        ax3.set_title('Model Agreement', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Performance Summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # Calculate detection stats for each method
        lstmae_pred = lstmae_scores > 1.0
        if_pred = if_scores > 1.0
        ensemble_pred = ensemble_scores > 1.0
        
        def calc_metrics(pred, true):
            tp = np.sum((pred == 1) & (true == 1))
            fp = np.sum((pred == 1) & (true == 0))
            tn = np.sum((pred == 0) & (true == 0))
            fn = np.sum((pred == 0) & (true == 1))
            
            acc = (tp + tn) / len(true) if len(true) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            return acc, prec, rec
        
        lstmae_metrics = calc_metrics(lstmae_pred, labels)
        if_metrics = calc_metrics(if_pred, labels)
        ensemble_metrics = calc_metrics(ensemble_pred, labels)
        
        summary_text = f"""
        Performance Comparison
        {'='*35}
        
        LSTM-AE:
          Acc: {lstmae_metrics[0]:.3f}
          Prec: {lstmae_metrics[1]:.3f}
          Rec: {lstmae_metrics[2]:.3f}
        
        Isolation Forest:
          Acc: {if_metrics[0]:.3f}
          Prec: {if_metrics[1]:.3f}
          Rec: {if_metrics[2]:.3f}
        
        Ensemble:
          Acc: {ensemble_metrics[0]:.3f}
          Prec: {ensemble_metrics[1]:.3f}
          Rec: {ensemble_metrics[2]:.3f}
        
        Samples: {len(labels)}
        Anomalies: {np.sum(labels)}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                 verticalalignment='center')
        
        fig.suptitle('Ensemble Detection Analysis', fontsize=14, fontweight='bold')        
    
    def save_plot(self):
        if not hasattr(self, 'viz_figure'):
            messagebox.showwarning("Warning", "No plot to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")],
            initialfile=f"lstmif_plot_{datetime.now().strftime('%y%m%d_%H%M')}"
        )
        
        if file_path:
            self.viz_figure.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot saved:\n{file_path}")
    
    def export_results_csv(self):
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"lstmif_results_{datetime.now().strftime('%y%m%d_%H%M')}.csv"
        )
        
        if file_path:
            export_data = []
            
            # Detection method 확인
            is_ensemble = any(r.get('detection_method') == 'ensemble' for r in self.current_results)
            
            for result in self.current_results:
                # 기본 필드 (항상 포함)
                row = {
                    'file': result.get('file', 'N/A'),
                    'detection_method': result.get('detection_method', 'unknown'),
                    'is_anomaly': result.get('is_anomaly', False),
                    'vacuum_range': result.get('vacuum_range', 'N/A'),
                    'vacuum_value': result.get('vacuum_value', 0),
                    'confidence': result.get('confidence', 0)
                }
                
                # Detection method별 필드
                if result.get('detection_method') == 'ensemble':
                    row.update({
                        'ensemble_score': result.get('ensemble_score', 0),
                        'ensemble_strategy': result.get('ensemble_strategy', 'N/A'),
                        'lstmae_normalized_score': result.get('lstmae_normalized_score', 0),
                        'if_normalized_score': result.get('if_normalized_score', 0),
                        'lstmae_weight': result.get('lstmae_weight', 0),
                        'if_weight': result.get('if_weight', 0)
                    })
                    
                    # LSTM-AE 세부 정보
                    if 'lstmae_result' in result:
                        lstmae = result['lstmae_result']
                        row.update({
                            'lstmae_anomaly_score': lstmae.get('anomaly_score', 0),
                            'lstmae_threshold': lstmae.get('threshold', 0),
                            'lstmae_adjusted_threshold': lstmae.get('adjusted_threshold', 0),
                            'mse': lstmae.get('mse', 0),
                            'weighted_error': lstmae.get('weighted_error', 0),
                            'peak_time': lstmae.get('peak_time', 0)
                        })
                        
                        if 'anomaly_sensors' in lstmae:
                            row['anomaly_sensors'] = ', '.join(lstmae.get('anomaly_sensors', []))
                    
                    # IF 세부 정보
                    if 'if_result' in result:
                        if_res = result['if_result']
                        row.update({
                            'if_anomaly_probability': if_res.get('anomaly_probability', 0),
                            'if_raw_score': if_res.get('raw_score', 0)
                        })
                
                else:  # LSTM-AE only
                    row.update({
                        'anomaly_score': result.get('anomaly_score', 0),
                        'threshold': result.get('threshold', 0),
                        'adjusted_threshold': result.get('adjusted_threshold', 0),
                        'mse': result.get('mse', 0),
                        'weighted_error': result.get('weighted_error', 0),
                        'peak_time': result.get('peak_time', 0)
                    })
                    
                    if 'anomaly_sensors' in result:
                        row['anomaly_sensors'] = ', '.join(result.get('anomaly_sensors', []))
                
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            
            # 컬럼 순서 정리
            base_cols = ['file', 'detection_method', 'is_anomaly', 'vacuum_range', 'vacuum_value', 'confidence']
            other_cols = [col for col in df.columns if col not in base_cols]
            df = df[base_cols + other_cols]
            
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            messagebox.showinfo("Success", f"Results exported:\n{file_path}\n\nTotal rows: {len(export_data)}")
    
    def export_statistics(self):
        if self.model is None:
            messagebox.showwarning("Warning", "No trained model available.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"lstmif_stats_{datetime.now().strftime('%y%m%d_%H%M')}.csv"
        )
        
        if file_path:
            stats_data = [{
                'Model_ID': self.model.model_id,
                'Timestamp': self.model.created_at.strftime('%y%m%d_%H%M') if self.model.created_at else 'N/A',
                'Model_Type': 'Unified',
                'Created': self.model.created_at.isoformat() if self.model.created_at else 'N/A',
                'Base_Threshold': self.model.threshold,
                'Training_Time_s': self.model.training_time,
                'Best_Epoch': self.model.best_epoch,
                'Train_Mean': self.model.train_stats.get('mean', 0),
                'Train_Std': self.model.train_stats.get('std', 0),
                'Train_Q95': self.model.train_stats.get('q95', 0),
                'Val_Mean': self.model.val_stats.get('mean', 0),
                'Val_Std': self.model.val_stats.get('std', 0),
                'Val_Q95': self.model.val_stats.get('q95', 0),
                'Encoder_Units': str(self.model.encoder_units),
                'Latent_Dim': self.model.latent_dim,
                'Dropout': self.model.dropout,
                'Recurrent_Dropout': self.model.recurrent_dropout,
                'LR': self.model.lr,
                'Bidirectional': self.model.use_bidirectional
            }]
            
            for range_name, adj in self.model.vacuum_range_thresholds.items():
                stats_data[0][f'{range_name}_Adjustment'] = adj
                stats_data[0][f'{range_name}_Threshold'] = self.model.threshold * adj
            
            df = pd.DataFrame(stats_data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Success", f"Statistics exported:\n{file_path}")
    
    def generate_full_report(self):
        if not self.current_results:
            messagebox.showwarning("Warning", "No detection results available.")
            return
        
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        
        if self.training_data_dir:
            report_dir = self.training_data_dir / "LSTM_IF" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"report_{timestamp}.html"
        else:
            report_path = Path.cwd() / f"lstmif_report_{timestamp}.html"
        
        # Detection method 확인 - 안전하게
        detection_method = self.current_results[0].get('detection_method', 'unknown') if self.current_results else 'unknown'
        is_ensemble = detection_method == 'ensemble'
        
        # 통계 계산
        total_files = len(self.current_results)
        anomalies = sum(1 for r in self.current_results if r.get('is_anomaly', False))
        normals = total_files - anomalies
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>LSTM-{'IF Ensemble' if is_ensemble else 'AE'} Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background-color: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; 
                            box-shadow: 0 0 20px rgba(0,0,0,0.1); border-radius: 10px; }}
                h1 {{ color: #1976D2; border-bottom: 3px solid #1976D2; padding-bottom: 10px; }}
                h2 {{ color: #2E7D32; border-bottom: 2px solid #2E7D32; padding-bottom: 5px; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #1976D2; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .anomaly {{ color: #C62828; font-weight: bold; }}
                .normal {{ color: #2E7D32; font-weight: bold; }}
                .summary {{ background-color: #E8F5E9; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                .ensemble-badge {{ background-color: #F57C00; color: white; padding: 5px 10px; 
                                 border-radius: 5px; font-weight: bold; display: inline-block; }}
                .method-info {{ background-color: #FFF3E0; padding: 15px; margin: 15px 0; border-radius: 5px; 
                              border-left: 4px solid #F57C00; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>LSTM-{'IF Ensemble' if is_ensemble else 'AE'} Analysis Report</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                {'<span class="ensemble-badge">🔥 ENSEMBLE DETECTION</span>' if is_ensemble else '<span class="ensemble-badge" style="background-color: #1976D2;">🔍 LSTM-AE DETECTION</span>'}
                
                <div class="summary">
                    <h3>Executive Summary</h3>
                    <p><strong>Model:</strong> {self.model.model_id if self.model else 'N/A'}</p>
                    <p><strong>Detection Method:</strong> {'Ensemble (LSTM-AE + Isolation Forest)' if is_ensemble else 'LSTM-AE Only'}</p>
                    <p><strong>Total Files:</strong> {total_files}</p>
                    <p><strong>Anomalies:</strong> {anomalies} ({anomalies/total_files*100:.1f}%)</p>
                    <p><strong>Normal:</strong> {normals} ({normals/total_files*100:.1f}%)</p>
                </div>
        """
        
        # Detection Method 설명
        if is_ensemble:
            first_result = self.current_results[0]
            html_content += f"""
                <div class="method-info">
                    <h3>🔥 Ensemble Configuration</h3>
                    <p><strong>Strategy:</strong> {first_result.get('ensemble_strategy', 'N/A')}</p>
                    <p><strong>LSTM-AE Weight:</strong> {first_result.get('lstmae_weight', 0):.2f}</p>
                    <p><strong>Isolation Forest Weight:</strong> {first_result.get('if_weight', 0):.2f}</p>
                    <p><em>The ensemble combines temporal pattern detection (LSTM-AE) with feature-based anomaly detection (Isolation Forest) 
                       for more robust and accurate results.</em></p>
                </div>
            """
        
        # Model Information
        html_content += """
                <h2>Model Information</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
        """
        
        if self.model:
            html_content += f"""
                    <tr><td>Model ID</td><td>{self.model.model_id}</td></tr>
                    <tr><td>Architecture</td><td>Encoder: {self.model.encoder_units}, Latent: {self.model.latent_dim}</td></tr>
                    <tr><td>Base Threshold</td><td>{self.model.threshold:.6f}</td></tr>
                    <tr><td>Training Time</td><td>{self.model.training_time:.1f}s</td></tr>
                    <tr><td>Best Epoch</td><td>{self.model.best_epoch}</td></tr>
                    <tr><td>Train MSE</td><td>{self.model.train_stats.get('mean', 0):.6f}</td></tr>
                    <tr><td>Val MSE</td><td>{self.model.val_stats.get('mean', 0):.6f}</td></tr>
            """
        
        html_content += """
                </table>
                
                <h2>Detection Results</h2>
                <table>
        """
        
        # 테이블 헤더 - detection method에 따라 다르게
        if is_ensemble:
            html_content += """
                    <tr>
                        <th>File</th>
                        <th>Vacuum Range</th>
                        <th>Status</th>
                        <th>Ensemble Score</th>
                        <th>LSTM-AE Score</th>
                        <th>IF Score</th>
                        <th>Confidence</th>
                    </tr>
            """
        else:
            html_content += """
                    <tr>
                        <th>File</th>
                        <th>Vacuum Range</th>
                        <th>Status</th>
                        <th>Anomaly Score</th>
                        <th>Threshold</th>
                        <th>Confidence</th>
                    </tr>
            """
        
        # 결과 행
        for result in self.current_results:
            status_class = 'anomaly' if result['is_anomaly'] else 'normal'
            status_text = '🔴 ANOMALY' if result['is_anomaly'] else '🟢 NORMAL'
            
            if is_ensemble:
                html_content += f"""
                        <tr>
                            <td>{result.get('file', 'N/A')}</td>
                            <td>{result.get('vacuum_range', 'N/A')}</td>
                            <td class="{status_class}">{status_text}</td>
                            <td>{result.get('ensemble_score', 0):.4f}</td>
                            <td>{result.get('lstmae_normalized_score', 0):.4f}</td>
                            <td>{result.get('if_normalized_score', 0):.4f}</td>
                            <td>{result.get('confidence', 0)*100:.1f}%</td>
                        </tr>
                """
            else:
                html_content += f"""
                        <tr>
                            <td>{result.get('file', 'N/A')}</td>
                            <td>{result.get('vacuum_range', 'N/A')}</td>
                            <td class="{status_class}">{status_text}</td>
                            <td>{result.get('anomaly_score', 0):.6f}</td>
                            <td>{result.get('adjusted_threshold', 0):.6f}</td>
                            <td>{result.get('confidence', 0)*100:.1f}%</td>
                        </tr>
                """
        
        # Vacuum Range별 통계
        html_content += """
                </table>
                
                <h2>Statistics by Vacuum Range</h2>
                <table>
                    <tr>
                        <th>Vacuum Range</th>
                        <th>Total</th>
                        <th>Anomalies</th>
                        <th>Normal</th>
                        <th>Anomaly Rate</th>
                    </tr>
        """
        
        # Vacuum range별 집계
        range_stats = {}
        for result in self.current_results:
            vr = result.get('vacuum_range', 'UNKNOWN')
            if vr not in range_stats:
                range_stats[vr] = {'total': 0, 'anomaly': 0, 'normal': 0}
            
            range_stats[vr]['total'] += 1
            if result['is_anomaly']:
                range_stats[vr]['anomaly'] += 1
            else:
                range_stats[vr]['normal'] += 1
        
        for vr, stats in sorted(range_stats.items()):
            anomaly_rate = stats['anomaly'] / stats['total'] * 100 if stats['total'] > 0 else 0
            html_content += f"""
                    <tr>
                        <td>{vr}</td>
                        <td>{stats['total']}</td>
                        <td class="anomaly">{stats['anomaly']}</td>
                        <td class="normal">{stats['normal']}</td>
                        <td>{anomaly_rate:.1f}%</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # ✅ 파일 저장 및 브라우저 열기를 한 번만 실행
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.log(f"Report saved: {report_path.name}")
            messagebox.showinfo("Success", f"Report generated:\n{report_path}")
            
            # 브라우저에서 열기
            try:
                import webbrowser
                webbrowser.open(str(report_path))
            except Exception as e:
                self.log(f"Could not open browser: {str(e)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report:\n{str(e)}")

    def delete_selected_model(self):
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to delete.")
            return
        
        item = self.model_tree.item(selection[0])
        timestamp = item['values'][0]
        
        if messagebox.askyesno("Confirm Delete", f"Delete model {timestamp}?"):
            if timestamp in self.loaded_models_info:
                model_path = self.loaded_models_info[timestamp]['path']
                
                try:
                    model_path.unlink()
                    
                    stem = model_path.stem
                    metadata_path = model_path.parent / f"{stem}_metadata.pkl"
                    scalers_path = model_path.parent / f"{stem}_scalers.pkl"
                    
                    if metadata_path.exists():
                        metadata_path.unlink()
                    if scalers_path.exists():
                        scalers_path.unlink()
                    
                    self.log(f"Deleted model: {timestamp}")
                    self.load_existing_models()
                    messagebox.showinfo("Success", "Model deleted successfully.")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete model:\n{str(e)}")
    
    def load_all_data(self, folder_path):
        csv_files = list(folder_path.glob("*.csv"))
        all_data = []
        
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if len(df) >= 7000:
                    data = df.iloc[:7000, 1:8].values.astype(np.float32)
                    all_data.append(data)
            except Exception as e:
                self.log(f"Skipped {file.name}: {str(e)}")
                continue
        
        return all_data
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.train_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.train_log.see(tk.END)
        self.train_log.update()
        self.status_var.set(message[:70])
    
    def run(self):
        self.root.mainloop()

# ================== Main ==================
if __name__ == "__main__":
    print("="*70)
    print("Cryo Pump AI Anomaly Detection for EV Tech Team")
    print("GPU Optimized for RTX 2080Ti")
    print("Single Model for All Vacuum Ranges")
    print("Advanced Training Controls")
    print("Created by SeJun Kang")
    print("="*70)
    
    gpu_available = GPUManager.setup_gpu()
    
    app = EnhancedLSTMAEGUI(gpu_available)
    app.run()
