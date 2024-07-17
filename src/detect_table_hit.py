import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import librosa
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import threading
import queue
import random

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# プロジェクトのルートディレクトリを取得
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# パラメータ設定
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DURATION = 3  # 3秒のサンプルを使用
CONFIDENCE_THRESHOLD = 0.3  # 検出の信頼度閾値
DISPLAY_TIME = 3  # 画像表示時間（秒）
WARMUP_TIME = 5  # ウォームアップ時間（秒）を追加

# 画像のパスを設定
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'images')
REGULAR_IMAGES = ['image 1.png', 'image 2.png', 'image 3.png', 'image 4.png']
RARE_IMAGES = ['image 5.png', 'image 6.png']

# モデルとスケーラーの読み込み
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'svm_model.joblib')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.joblib')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("モデルとスケーラーを正常に読み込みました。")
except Exception as e:
    logging.error(f"モデルまたはスケーラーの読み込みに失敗しました: {e}")
    raise

def select_image():
    if random.random() < 0.05:  # 5%の確率で希少な画像を選択
        return os.path.join(IMAGE_DIR, random.choice(RARE_IMAGES))
    else:
        return os.path.join(IMAGE_DIR, random.choice(REGULAR_IMAGES))

def extract_features(audio_data, sr):
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sr)
        
        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1)
        ])
        
        return features
    except Exception as e:
        logging.error(f"特徴量抽出中にエラーが発生しました: {e}")
        raise

def detect_table_hit(audio_buffer):
    try:
        audio_data = audio_buffer.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        logging.debug(f"Normalized audio data max: {np.max(np.abs(audio_data)):.6f}")

        # 特徴量抽出
        features = extract_features(audio_data, RATE)
        
        # 特徴量のスケーリング
        scaled_features = scaler.transform(features.reshape(1, -1))
        
        # モデルによる予測
        prediction = model.predict(scaled_features)
        
        # 予測結果とconfidence（確信度）を取得
        is_table_hit = prediction[0] == 'desk_punch'
        confidence = model.decision_function(scaled_features)[0]
        
        logging.info(f"Prediction: {'Table hit' if is_table_hit else 'Normal sound'}, Confidence: {confidence:.2f}")
        
        return is_table_hit and abs(confidence) > CONFIDENCE_THRESHOLD
    except Exception as e:
        logging.error(f"台パン検出中にエラーが発生しました: {e}")
        return False

def display_image():
    image_path = select_image()
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(DISPLAY_TIME)
    plt.close()
    logging.info(f"表示した画像: {os.path.basename(image_path)}")

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

audio_queue = queue.Queue()

# PyAudioストリームを開始
try:
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)
    logging.info("音声ストリームを正常に開始しました。")
except Exception as e:
    logging.error(f"音声ストリームの開始に失敗しました: {e}")
    raise

stream.start_stream()

BUFFER_SIZE = RATE * DURATION
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.int16)

logging.info("台パン検知を開始します。Ctrl+Cで停止します。")

# ウォームアップ期間の追加
logging.info(f"ウォームアップ中... {WARMUP_TIME}秒お待ちください。")
start_time = time.time()

try:
    while True:
        current_time = time.time()
        
        # ウォームアップ期間中はデータを収集するだけで検出は行わない
        if current_time - start_time < WARMUP_TIME:
            while not audio_queue.empty():
                data = audio_queue.get()
                new_data = np.frombuffer(data, dtype=np.int16)
                audio_buffer = np.roll(audio_buffer, -len(new_data))
                audio_buffer[-len(new_data):] = new_data
            time.sleep(0.1)
            continue

        # キューからデータを取得
        while not audio_queue.empty():
            data = audio_queue.get()
            new_data = np.frombuffer(data, dtype=np.int16)
            audio_buffer = np.roll(audio_buffer, -len(new_data))
            audio_buffer[-len(new_data):] = new_data

        if detect_table_hit(audio_buffer):
            logging.info(f"台パンを検出しました: {time.time():.2f}")
            
            # 画像を表示
            display_image()
            
            logging.info("検出を再開します。")
        
        time.sleep(0.1)  # CPUの使用率を下げるための小さな遅延

except KeyboardInterrupt:
    logging.info("ユーザーによってプログラムが停止されました。")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    logging.info("音声ストリームを閉じました。")

logging.info("プログラムが正常に終了しました。")