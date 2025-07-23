import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta

# Load model dan scaler
@st.cache_resource
def load_artifacts():
    model = load_model('model_lstm_solana_final_fixed.h5', compile=False)
    feature_scaler = joblib.load('scaler_lstm_solana_features_final.pkl')
    target_scaler = joblib.load('scaler_lstm_solana_target_final.pkl')
    return model, feature_scaler, target_scaler

model, feature_scaler, target_scaler = load_artifacts()

# Header aplikasi
st.title("📈 Prediksi Harga Solana (SOL) 6 Bulan ke Depan")
st.markdown("Prediksi dilakukan menggunakan model LSTM dengan fitur teknikal seperti MA, EMA, RSI, MACD, dan Price Range.")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload file CSV harga Solana (opsional):", type=["csv"])

# Gunakan default jika tidak upload
if uploaded_file is not None:
    st.success("✅ Menggunakan data yang diupload.")
    df = pd.read_csv(uploaded_file)
else:
    st.info("📌 Tidak ada file diupload, menggunakan data default.")
    df = pd.read_csv('solana_final.csv')

# Preprocessing
try:
    df = df.drop(index=[0, 1]).reset_index(drop=True)
except:
    pass

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# Pilih kolom penting dan ubah ke numerik
kolom_numerik = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in kolom_numerik:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().reset_index(drop=True)

# Tambahkan indikator teknikal
df['MA_14'] = df['Close'].rolling(window=14).mean()
df['EMA_14'] = df['Close'].ewm(span=14).mean()
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))
df['MACD'] = df['EMA_14'] - df['Close'].ewm(span=26).mean()
df['Price_Range'] = df['High'] - df['Low']

df = df.dropna().reset_index(drop=True)

# Fitur dan normalisasi
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_14', 'EMA_14', 'RSI_14', 'MACD', 'Price_Range']
scaled_features = feature_scaler.transform(df[features])
window_size = 60

# Rolling prediction
input_seq = scaled_features[-window_size:].copy()
future_days = 180
predictions = []

for _ in range(future_days):
    reshaped = input_seq.reshape(1, window_size, len(features))
    pred = model.predict(reshaped, verbose=0)[0][0]
    dummy_next = input_seq[-1].copy()
    dummy_next[features.index('Close')] = pred
    dummy_next += np.random.normal(0, 0.005, dummy_next.shape)
    input_seq = np.vstack([input_seq[1:], dummy_next])
    predictions.append(pred)

# Inverse transform
predicted_prices = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
future_dates = pd.date_range(start=df['Date'].iloc[-1] + timedelta(days=1), periods=future_days)

# Plot hasil
st.subheader("📊 Grafik Prediksi Harga")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Close'], label='Harga Historis', color='blue')
ax.plot(future_dates, predicted_prices, label='Prediksi 6 Bulan ke Depan', color='green')
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga (USD)")
ax.set_title("Prediksi Harga Solana 6 Bulan ke Depan")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Prediksi per tanggal
st.subheader("🔍 Prediksi Berdasarkan Tanggal")
date_option = st.date_input("Pilih tanggal prediksi (maks 6 bulan ke depan):", 
                            min_value=future_dates[0].date(), 
                            max_value=future_dates[-1].date())

if date_option in future_dates.date:
    index = list(future_dates.date).index(date_option)
    value = predicted_prices[index][0]
    st.success(f"🎯 Prediksi harga SOL pada {date_option} adalah **${value:.2f}**")
else:
    st.warning("Tanggal di luar jangkauan prediksi.")
