# Google Colab対応LLM学習コース

このリポジトリは、Google Colab環境でLLM（Large Language Model）を学習するための包括的なコースです。

## 🚀 クイックスタート

1. **Google Colabを開く**
   - [Google Colab](https://colab.research.google.com/)にアクセス
   - 新しいノートブックを作成

2. **ノートブックを読み込む**
   - このリポジトリからノートブックを選択
   - または、GitHubから直接読み込み

3. **実行開始**
   - ランタイム > ランタイムのタイプを変更 > GPUを選択
   - セルを順番に実行

## 📚 学習コース構成

### 基礎編
- **`notebooks/02_colab_llm_basics.ipynb`** - LLM基礎実習
  - 環境セットアップ
  - 基本的なテキスト生成
  - パラメータ調整
  - プロンプトエンジニアリング
  - Few-shot学習

### 応用編
- **`notebooks/06_colab_advanced_fine_tuning.ipynb`** - 高度なファインチューニング
  - カスタムデータセットでのファインチューニング
  - 異なるファインチューニング手法の比較
  - ハイパーパラメータ最適化
  - 実用的なアプリケーション例

- **`notebooks/07_colab_japanese_llm.ipynb`** - 日本語LLM実習
  - 日本語特化モデルの使用
  - 日本語プロンプトエンジニアリング
  - 日本語テキストの前処理と最適化
  - 日本語での実用的なアプリケーション

- **`notebooks/08_colab_competition_advanced.ipynb`** - コンペティション高度テクニック
  - 高度なアンサンブル手法
  - データ拡張技術
  - 自動ハイパーパラメータ最適化
  - パフォーマンス分析と可視化

### 演習問題
- **`exercises/03_colab_basic_exercises.py`** - 基礎演習問題
- **`exercises/04_colab_advanced_exercises.py`** - 高度演習問題

### 実践プロジェクト
- **`projects/03_colab_sentiment_analysis_project.py`** - 感情分析プロジェクト
- **`projects/04_colab_chatbot_project.py`** - チャットボットプロジェクト

## 🛠️ 環境要件

### 必須ライブラリ
```python
# 基本ライブラリ
torch
transformers
datasets
accelerate
evaluate
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm

# 日本語処理（日本語LLM用）
fugashi
ipadic
mecab-python3
jaconv

# 高度な機能
peft
bitsandbytes
optuna
plotly
```

### 推奨設定
- **ランタイム**: GPU（T4以上推奨）
- **メモリ**: 高メモリ（必要に応じて）
- **ディスク**: 標準（モデルキャッシュ用）

## 📖 学習の進め方

### 1. 基礎から始める
1. `02_colab_llm_basics.ipynb`で基本概念を学習
2. `03_colab_basic_exercises.py`で演習問題を解く
3. 基本的なテキスト生成をマスター

### 2. 応用技術を習得
1. `06_colab_advanced_fine_tuning.ipynb`でファインチューニングを学習
2. `07_colab_japanese_llm.ipynb`で日本語LLMを体験
3. `08_colab_competition_advanced.ipynb`で高度なテクニックを習得

### 3. 実践プロジェクトで応用
1. `03_colab_sentiment_analysis_project.py`で感情分析アプリを開発
2. `04_colab_chatbot_project.py`でチャットボットを構築
3. 実際のビジネス課題に適用

## ⚠️ Colab環境での制限と対処法

### メモリ制限
- **問題**: 大きなモデルでメモリ不足
- **対処法**: 
  - 定期的にランタイムを再起動
  - 軽量なモデルを使用
  - バッチサイズを小さく設定

### 実行時間制限
- **問題**: 長時間の処理でタイムアウト
- **対処法**:
  - 処理を分割して実行
  - 中間結果を保存
  - 定期的にチェックポイントを作成

### ファイル保存制限
- **問題**: 生成されたファイルが失われる
- **対処法**:
  - Google Driveに定期的に保存
  - 重要な結果は即座にダウンロード
  - バックアップを取る

## 🔧 トラブルシューティング

### よくある問題と解決法

#### 1. モデルの読み込みエラー
```python
# エラー例
RuntimeError: CUDA out of memory

# 解決法
# 1. ランタイムを再起動
# 2. より軽量なモデルを使用
# 3. バッチサイズを小さく設定
```

#### 2. ライブラリのインストールエラー
```python
# エラー例
ModuleNotFoundError: No module named 'transformers'

# 解決法
!pip install transformers
# または
!pip install -r requirements.txt
```

#### 3. メモリ不足エラー
```python
# エラー例
torch.cuda.OutOfMemoryError

# 解決法
# 1. メモリをクリア
torch.cuda.empty_cache()
# 2. ランタイムを再起動
# 3. より軽量な設定を使用
```

## 📊 パフォーマンス最適化

### GPU使用の最適化
```python
# GPU使用状況の確認
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

### メモリ効率の向上
```python
# メモリクリア
torch.cuda.empty_cache()

# 軽量化設定
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 半精度
    device_map="auto"
)
```

## 🎯 学習目標

### 基礎レベル
- [ ] LLMの基本概念を理解
- [ ] 基本的なテキスト生成ができる
- [ ] パラメータ調整ができる
- [ ] プロンプトエンジニアリングができる

### 中級レベル
- [ ] カスタムデータセットでファインチューニングができる
- [ ] 日本語LLMを使用できる
- [ ] 実用的なアプリケーションを開発できる
- [ ] パフォーマンスを評価できる

### 上級レベル
- [ ] 高度なアンサンブル手法を実装できる
- [ ] データ拡張技術を活用できる
- [ ] ハイパーパラメータ最適化ができる
- [ ] コンペティションで成果を出せる

## 🤝 コミュニティ

### 質問・相談
- GitHub Issuesで質問
- ディスカッションで議論
- プルリクエストで改善提案

### 貢献
- バグレポート
- 機能追加
- ドキュメント改善
- 学習コンテンツの追加

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞

- Hugging Face Transformers
- Google Colab
- オープンソースコミュニティ
- 学習者の皆様

---

**Happy Learning! 🚀**

Google Colab環境でLLMの学習を楽しんでください。質問やフィードバックがあれば、お気軽にお声がけください。
