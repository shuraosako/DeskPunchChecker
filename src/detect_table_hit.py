import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import time
import librosa
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import threading
import queue
import random
import tkinter as tk
import cv2
from collections import deque
import datetime
import wave
import tempfile
import subprocess

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# プロジェクトのルートディレクトリを取得
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# パラメータ設定
CHUNK = 1024
FORMAT = pyaudio.paInt16 # 16ビットの整数形式
CHANNELS = 1 # 音声チャンネル数
RATE = 44100 # サンプリング周波数
DURATION = 3  # 3秒のサンプルを使用
CONFIDENCE_THRESHOLD = 0.3  # 検出の信頼度閾値
DISPLAY_TIME = 3  # 画像表示時間（秒）
WARMUP_TIME = 5  # ウォームアップ時間（秒）を追加

# ビデオ関連の設定
VIDEO_BUFFER_SECONDS = 3  # ビデオバッファの秒数 
AUDIO_BUFFER_SECONDS = 3  # 音声バッファの秒数
FPS = 20  # ビデオのフレームレート
VIDEO_OUTPUT_DIR = r"C:\Users\81809\Documents\学校\ハッカソン\HACKU2024\test-test\HACKU2024-Final -2\ビデオデータ"
VIDEO_WIDTH = 640  # ビデオ幅
VIDEO_HEIGHT = 480  # ビデオ高さ

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(VIDEO_OUTPUT_DIR):
    os.makedirs(VIDEO_OUTPUT_DIR)

# 画像のパスを設定
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'images')
REGULAR_IMAGES = ['image 1.png', 'image 2.png', 'image 3.png']
RARE_IMAGES = ['image 4.png', 'image 5.png']

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
    if random.random() < 0.3:  # 30%の確率で希少な画像を選択
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
        audio_data = audio_buffer.astype(np.float32) / 32768.0
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

def display_image_topmost():
    image_path = select_image()
    root = tk.Tk()
    root.attributes('-topmost', True)  # ウィンドウを最前面に設定
    root.overrideredirect(True)  # ウィンドウの枠を削除

    img = Image.open(image_path)
    photo = ImageTk.PhotoImage(img)
    
    label = tk.Label(root, image=photo)
    label.pack()

    def close_window(event):
        root.destroy()

    root.bind('<Button-1>', close_window)  # クリックでウィンドウを閉じる

    root.after(int(DISPLAY_TIME * 1000), root.destroy)  # DISPLAY_TIME秒後にウィンドウを閉じる

    root.mainloop()
    
    logging.info(f"表示した画像: {os.path.basename(image_path)}")

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

# 音声データをWAVEファイルとして保存する関数
def save_audio_buffer(audio_buffer_data):
    # 一時ファイルに音声を保存
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_wav.close()
    
    try:
        with wave.open(temp_wav.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit なので 2 バイト
            wf.setframerate(RATE)
            wf.writeframes(audio_buffer_data)
        
        return temp_wav.name
    except Exception as e:
        logging.error(f"音声ファイルの保存中にエラーが発生しました: {e}")
        if os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)
        return None

# ビデオと音声を結合する関数
def combine_video_audio(video_path, audio_path, output_path):
    try:
        # FFmpegを使用して、ビデオと音声を結合
        command = [
            'ffmpeg',
            '-i', video_path,  # ビデオファイル
            '-i', audio_path,  # 音声ファイル
            '-c:v', 'copy',    # ビデオコーデックをコピー
            '-c:a', 'aac',     # 音声コーデックをAACに設定
            '-strict', 'experimental',
            '-y',              # 既存のファイルを上書き
            output_path
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"ビデオと音声を結合しました: {output_path}")
        
        # 一時ファイルを削除
        os.unlink(video_path)
        os.unlink(audio_path)
        
        return True
    except Exception as e:
        logging.error(f"ビデオと音声の結合中にエラーが発生しました: {e}")
        return False

# ビデオフレームを保存する関数
def save_video_buffer(frame_buffer, audio_buffer_data):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_video_{timestamp}.mp4")
    final_output_path = os.path.join(VIDEO_OUTPUT_DIR, f"table_hit_{timestamp}.mp4")
    
    # 音声データを一時ファイルに保存
    audio_path = save_audio_buffer(audio_buffer_data)
    if not audio_path:
        logging.error("音声データの保存に失敗しました。ビデオのみ保存します。")
        # 音声なしでビデオだけ保存
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(final_output_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
        try:
            for frame in frame_buffer:
                out.write(frame)
            logging.info(f"ビデオを保存しました（音声なし）: {final_output_path}")
        finally:
            out.release()
        return
    
    # ビデオを一時ファイルに保存
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    
    try:
        # バッファ内のすべてのフレームを書き込む
        for frame in frame_buffer:
            out.write(frame)
        
        out.release()
        
        # ビデオと音声を結合
        if combine_video_audio(temp_video_path, audio_path, final_output_path):
            logging.info(f"音声付きビデオを保存しました: {final_output_path}")
        else:
            logging.warning("ビデオと音声の結合に失敗しました。")
    except Exception as e:
        logging.error(f"ビデオの保存中にエラーが発生しました: {e}")
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        if os.path.exists(audio_path):
            os.unlink(audio_path)
    finally:
        if out.isOpened():
            out.release()

# ビデオキャプチャとバッファリングのためのスレッド関数
def video_capture_thread():
    global is_running
    
    # Webカメラを初期化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("カメラのオープンに失敗しました。")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    
    # フレームバッファを初期化（最大サイズはFPS * VIDEO_BUFFER_SECONDS）
    frame_buffer = deque(maxlen=FPS * VIDEO_BUFFER_SECONDS)
    
    logging.info("ビデオキャプチャを開始しました。")
    
    try:
        while is_running:
            ret, frame = cap.read()
            if not ret:
                logging.error("フレームの読み取りに失敗しました。")
                break
                
            # フレームバッファに追加
            frame_buffer.append(frame)
            
            # バッファに保存する準備ができたことを通知
            video_frame_ready.set()
            
            # 台パンを検知したときにビデオを保存
            if should_save_video.is_set():
                # 現在の音声バッファのコピーを取得
                with audio_buffer_lock:
                    audio_data_copy = bytes(audio_buffer_raw)
                
                # ビデオを保存するための新しいスレッドを作成（メインスレッドをブロックしないため）
                save_thread = threading.Thread(target=save_video_buffer, args=(list(frame_buffer), audio_data_copy))
                save_thread.start()
                
                # フラグをリセット
                should_save_video.clear()
                
            # CPUの使用率を下げるための適切な遅延
            time.sleep(1/FPS)
    
    except Exception as e:
        logging.error(f"ビデオキャプチャ中にエラーが発生しました: {e}")
    
    finally:
        cap.release()
        logging.info("ビデオキャプチャを終了しました。")

audio_queue = queue.Queue()
video_frame_ready = threading.Event()
should_save_video = threading.Event()
is_running = True

# 音声バッファのサイズを計算
AUDIO_BUFFER_SIZE = RATE * AUDIO_BUFFER_SECONDS * 2  # 16ビット = 2バイト
audio_buffer_raw = bytearray(AUDIO_BUFFER_SIZE)
audio_buffer_lock = threading.Lock()
audio_buffer = np.zeros(RATE * DURATION, dtype=np.int16)  # 特徴抽出用バッファ

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

logging.info("台パン検知を開始します。Ctrl+Cで停止します。")

# ビデオキャプチャスレッドを開始
video_thread = threading.Thread(target=video_capture_thread)
video_thread.daemon = True
video_thread.start()

# ウォームアップ期間の追加
logging.info(f"ウォームアップ中... {WARMUP_TIME}秒お待ちください。")
start_time = time.time()

# 音声バッファに書き込むための位置（循環バッファ）
audio_write_pos = 0

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
                
                # 生の音声バッファにもデータを追加
                with audio_buffer_lock:
                    # 循環バッファとして扱う
                    data_len = len(data)
                    remaining = AUDIO_BUFFER_SIZE - audio_write_pos
                    if remaining >= data_len:
                        # バッファに十分な空きがある場合
                        audio_buffer_raw[audio_write_pos:audio_write_pos + data_len] = data
                        audio_write_pos = (audio_write_pos + data_len) % AUDIO_BUFFER_SIZE
                    else:
                        # バッファの終わりを超える場合は分割して書き込む
                        audio_buffer_raw[audio_write_pos:] = data[:remaining]
                        audio_buffer_raw[:data_len - remaining] = data[remaining:]
                        audio_write_pos = data_len - remaining
            
            time.sleep(0.1)
            continue

        # キューからデータを取得
        while not audio_queue.empty():
            data = audio_queue.get()
            new_data = np.frombuffer(data, dtype=np.int16)
            audio_buffer = np.roll(audio_buffer, -len(new_data))
            audio_buffer[-len(new_data):] = new_data
            
            # 生の音声バッファにもデータを追加
            with audio_buffer_lock:
                # 循環バッファとして扱う
                data_len = len(data)
                remaining = AUDIO_BUFFER_SIZE - audio_write_pos
                if remaining >= data_len:
                    # バッファに十分な空きがある場合
                    audio_buffer_raw[audio_write_pos:audio_write_pos + data_len] = data
                    audio_write_pos = (audio_write_pos + data_len) % AUDIO_BUFFER_SIZE
                else:
                    # バッファの終わりを超える場合は分割して書き込む
                    audio_buffer_raw[audio_write_pos:] = data[:remaining]
                    audio_buffer_raw[:data_len - remaining] = data[remaining:]
                    audio_write_pos = data_len - remaining

        if detect_table_hit(audio_buffer):
            logging.info(f"台パンを検出しました: {time.time():.2f}")
            
            # フレームバッファが準備できているか確認
            if video_frame_ready.is_set():
                # ビデオを保存するフラグを設定
                should_save_video.set()
                logging.info("台パン検出：ビデオバッファと音声バッファの保存を開始します。")
            
            # 別スレッドで画像表示を実行
            threading.Thread(target=display_image_topmost).start()
            
            # 少し待機して連続検出を防止
            time.sleep(1.0)
            
            logging.info("検出を再開します。")
        
        time.sleep(0.1)  # CPUの使用率を下げるための遅延

except KeyboardInterrupt:
    logging.info("ユーザーによってプログラムが停止されました。")

finally:
    is_running = False  # ビデオキャプチャスレッドを停止
    stream.stop_stream()
    stream.close()
    p.terminate()
    logging.info("音声ストリームを閉じました。")
    # ビデオスレッドの終了を待つ
    video_thread.join(timeout=2.0)
    logging.info("プログラムが正常に終了しました。")