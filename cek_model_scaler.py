from keras.models import load_model
import joblib

# Load model dan scaler
model = load_model("model_lstm_solana_final.h5")
scaler = joblib.load("scaler_lstm_solana_final.pkl")

# Tampilkan input shape model
print("Input shape model:", model.input_shape)

# Tampilkan jumlah fitur yang digunakan oleh scaler
print("Jumlah fitur yang digunakan oleh scaler:", scaler.n_features_in_)
