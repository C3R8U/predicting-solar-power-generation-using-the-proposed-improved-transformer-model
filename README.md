# predicting-solar-power-generation-using-the-proposed-improved-transformer-model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
data_path = 'C:/Users/86156/Desktop/Net generation from solar for United States Lower 48 (region) hourly - UTC time.csv'
df = pd.read_csv(data_path)

# Data Preprocessing
def preprocess_data(df):
    # Clean negative and zero values (treat as missing)
    df['megawatthours'] = df['megawatthours'].apply(lambda x: np.nan if x <= 0 else x)
    
    # Fill missing values with column mean
    for column in df.columns[1:]:
        col_mean = df[column].mean()
        df[column].fillna(col_mean, inplace=True)
    
    # Handle outliers using Z-score (|Z|>3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df[col] = np.where(z_scores > 3, df[col].median(), df[col])
    
    return df

# Feature Engineering
def create_features(df):
    # Parse datetime
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['season'] = df['month'] % 12 // 3 + 1
    
    # Selected features
    features = ['Temp', 'Sunshine', 'WindGustSpeed', 'Rainfall', 'Humidity', 
                'hour', 'day', 'month', 'season']
    target = 'megawatthours'
    
    # Normalization
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, features, target, scaler

# Create sliding window sequences
def create_sequences(data, features, target, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[features].iloc[i:i+window_size].values)
        y.append(data[target].iloc[i+window_size])
    return np.array(X), np.array(y)

# Transformer Model Components
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, inputs, enc_output, training):
        attn1 = self.mha1(inputs, inputs)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)
        attn2 = self.mha2(out1, enc_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

# Build Improved Transformer Model
def build_transformer_model(window_size, num_features, d_model=128, num_heads=8, 
                           dff=512, num_encoder_layers=2, num_decoder_layers=2, dropout_rate=0.1):
    # Encoder Input (Meteorological Factors)
    encoder_inputs = Input(shape=(window_size, num_features))
    x = Dense(d_model)(encoder_inputs)
    
    # Encoder Stack
    for _ in range(num_encoder_layers):
        x = TransformerEncoder(d_model, num_heads, dff, dropout_rate)(x)
    encoder_outputs = x
    
    # Decoder Input (Solar Power History)
    decoder_inputs = Input(shape=(window_size, 1))
    y = Dense(d_model)(decoder_inputs)
    
    # Decoder Stack
    for _ in range(num_decoder_layers):
        y = TransformerDecoder(d_model, num_heads, dff, dropout_rate)(y, encoder_outputs)
    decoder_outputs = y
    
    # Final Prediction
    final_output = decoder_outputs[:, -1, :]  # Last timestep
    final_output = Dense(64, activation='relu')(final_output)
    final_output = Dense(1)(final_output)
    
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=final_output)
    return model

# Main Pipeline
def main():
    # Preprocess data
    df = preprocess_data(df)
    df, features, target, scaler = create_features(df)
    
    # Create sequences
    X, y = create_sequences(df, features, target, window_size=24)
    
    # Split into solar and met features
    X_met = X[:, :, :-1]  # Meteorological factors
    X_solar = X[:, :, -1:]  # Solar power history
    
    # Train-test split (80-20)
    X_met_train, X_met_test, X_solar_train, X_solar_test, y_train, y_test = train_test_split(
        X_met, X_solar, y, test_size=0.2, shuffle=False
    )
    
    # Build model
    num_met_features = X_met.shape[-1]
    model = build_transformer_model(
        window_size=24,
        num_features=num_met_features,
        d_model=128,
        num_heads=8,
        dff=512,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        [X_met_train, X_solar_train],
        y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict([X_met_test, X_solar_test]).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Save model
    model.save('solar_forecast_transformer.h5')

if __name__ == "__main__":
    main()
