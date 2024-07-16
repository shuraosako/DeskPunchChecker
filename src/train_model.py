import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tqdm import tqdm
import joblib

# プロジェクトのルートディレクトリを取得
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 定数の設定
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
SAMPLE_RATE = 44100
DURATION = 3  # 3秒のサンプルを使用

def extract_features(file_path, duration=DURATION):
    # 音声ファイルを読み込み、特徴量を抽出する
    audio, sr = librosa.load(file_path, duration=duration, sr=SAMPLE_RATE)
    
    # 必要に応じて音声データをパディング
    if len(audio) < duration * sr:
        audio = np.pad(audio, (0, duration * sr - len(audio)))
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    
    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1),
        np.mean(contrast, axis=1),
        np.mean(tonnetz, axis=1)
    ])
    return features

def load_data():
    features = []
    labels = []
    
    # 環境音のデータ読み込み
    env_path = os.path.join(DATA_DIR, 'environment')
    for i in range(1, 51):  # レコーディング (1) から レコーディング (50) まで
        file_name = f"レコーディング ({i}).wav"
        file_path = os.path.join(env_path, file_name)
        if os.path.exists(file_path):
            try:
                feature = extract_features(file_path)
                features.append(feature)
                labels.append('environment')
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"File not found: {file_path}")

    # デスクパンチのデータ読み込み
    punch_path = os.path.join(DATA_DIR, 'desk_punch')
    for i in range(1, 258):  # レコーディング (1) から レコーディング (100) まで
        file_name = f"レコーディング ({i}).wav"
        file_path = os.path.join(punch_path, file_name)
        if os.path.exists(file_path):
            try:
                feature = extract_features(file_path)
                features.append(feature)
                labels.append('desk_punch')
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    return np.array(features), np.array(labels)

def train_model():
    print("データの読み込みと特徴量の抽出を開始します...")
    X, y = load_data()
    
    print("データの分割を行います...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("特徴量の正規化を行います...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("モデルの学習を開始します...")
    model = SVC(kernel='rbf', C=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("テストデータでの評価を行います...")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    
    print("モデルと正規化パラメータを保存します...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model, os.path.join(MODEL_DIR, 'svm_model.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    
    print("学習が完了しました。")

if __name__ == "__main__":
    train_model()