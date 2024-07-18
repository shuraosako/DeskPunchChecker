# 台パン検知システム

![DeskPunch](https://github.com/user-attachments/assets/4d67dae4-bac7-4350-a10e-281c87258762)

## 概要
このプロジェクトは、機械学習を用いて音声から「台パン」（テーブルを叩く音）を検出するシステムです。リアルタイムの音声入力を分析し、台パンが検出されると指定された画像を表示します。

## 機能
- リアルタイム音声分析
- 機械学習モデルを使用した台パン検出
- 台パン検出時の画像表示

## 必要条件
- Python 3.7以上
- pip（Pythonパッケージマネージャー）

## 使用方法

1. 学習用のデータを `data/environment` と `data/desk_punch` ディレクトリに配置します。

2. モデルを学習させます：
``python src/train_model.py``

3. 台パン検知を開始します：
``python src/detect_table_hit.py``

4. プログラムを終了するには、コンソールで Ctrl+C を押します。

## プロジェクト構造

```
project_root/
├── src/
│   ├── train_model.py
│   └── detect_table_hit.py
├── data/
│   ├── environment/
│   └── desk_punch/
├── models/
├── images/
│   └── image1.png
└── README.md
```

## カスタマイズ
- `src/detect_table_hit.py` 内の `CONFIDENCE_THRESHOLD` を調整して、検出感度を変更できます。
- `images/image1.png` を置き換えて、異なる画像を表示させることができます。

## トラブルシューティング
- 音声入力に問題がある場合は、マイクの設定を確認してください。
- 検出精度が低い場合は、より多くのトレーニングデータを追加するか、`train_model.py` 内のモデルパラメータを調整してください。


