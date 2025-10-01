# Transformerアーキテクチャの詳細

## 1. アーキテクチャの全体像

Transformerは、**Attention is All You Need**（2017年）で提案された、現在のLLMの基盤となるアーキテクチャです。

```
入力 → 埋め込み + 位置エンコーディング → エンコーダー層 → デコーダー層 → 出力
```

## 2. 主要コンポーネント

### 2.1 埋め込み（Embedding）
- **トークン埋め込み**: 各単語をベクトルに変換
- **位置埋め込み**: 単語の位置情報を追加
- **次元**: 通常512または768次元

### 2.2 マルチヘッドアテンション
```python
# 疑似コード
def multi_head_attention(Q, K, V, num_heads):
    # 1. 線形変換でQ, K, Vを生成
    Q = linear(Q)  # [batch, seq_len, d_model]
    K = linear(K)
    V = linear(V)
    
    # 2. ヘッド数に分割
    Q = reshape(Q, [batch, num_heads, seq_len, d_model//num_heads])
    K = reshape(K, [batch, num_heads, seq_len, d_model//num_heads])
    V = reshape(V, [batch, num_heads, seq_len, d_model//num_heads])
    
    # 3. スケールドドットプロダクトアテンション
    attention_scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
    attention_weights = softmax(attention_scores)
    output = attention_weights @ V
    
    # 4. ヘッドを結合
    output = reshape(output, [batch, seq_len, d_model])
    return output
```

### 2.3 フィードフォワードネットワーク
```python
def feed_forward(x):
    # 2層の全結合層
    hidden = relu(linear1(x))  # 通常4倍の次元
    output = linear2(hidden)
    return output
```

### 2.4 残差接続とレイヤー正規化
```python
def transformer_layer(x):
    # 1. マルチヘッドアテンション + 残差接続
    attn_output = multi_head_attention(x, x, x)
    x = layer_norm(x + attn_output)
    
    # 2. フィードフォワード + 残差接続
    ff_output = feed_forward(x)
    x = layer_norm(x + ff_output)
    
    return x
```

## 3. エンコーダーとデコーダー

### 3.1 エンコーダー
- **役割**: 入力シーケンスの表現を学習
- **構造**: マルチヘッドアテンション + フィードフォワード
- **自己注意**: 入力内の各位置が他の位置を参照

### 3.2 デコーダー
- **役割**: エンコーダーの出力から目標シーケンスを生成
- **構造**: マスク付き自己注意 + エンコーダー-デコーダー注意
- **マスク**: 未来の情報を見ないように制限

## 4. 位置エンコーディング

Transformerは並列処理のため、位置情報を明示的に追加する必要があります。

```python
def positional_encoding(seq_len, d_model):
    pos = arange(seq_len).reshape(-1, 1)
    i = arange(d_model).reshape(1, -1)
    
    # 偶数次元: sin
    pe[:, 0::2] = sin(pos / (10000 ** (2 * i[0::2] / d_model)))
    # 奇数次元: cos
    pe[:, 1::2] = cos(pos / (10000 ** (2 * i[1::2] / d_model)))
    
    return pe
```

## 5. 学習と推論

### 5.1 学習時
- **教師強制**: 正解シーケンスを入力として使用
- **並列処理**: 全位置を同時に処理
- **損失関数**: クロスエントロピー損失

### 5.2 推論時
- **自己回帰**: 生成したトークンを次の入力に使用
- **逐次処理**: トークンを一つずつ生成
- **ビームサーチ**: 複数の候補を保持

## 6. 実装のポイント

### 6.1 効率性
- **バッチ処理**: 複数のシーケンスを同時処理
- **メモリ効率**: 注意重みの計算を最適化
- **並列化**: GPUでの効率的な並列処理

### 6.2 安定性
- **勾配クリッピング**: 勾配爆発を防止
- **ドロップアウト**: 過学習を防止
- **レイヤー正規化**: 学習の安定化

## 7. 現代のLLMでの発展

### 7.1 スケーリング
- **パラメータ数**: 数億から数兆まで
- **データ量**: 数兆トークンの学習
- **計算資源**: 大規模なGPUクラスター

### 7.2 アーキテクチャの改良
- **RMSNorm**: レイヤー正規化の改良
- **SwiGLU**: 活性化関数の改良
- **RoPE**: 回転位置埋め込み

## 次のステップ

次の章では、実際にTransformerベースのモデルを使ってみる実践的な演習を行います。
