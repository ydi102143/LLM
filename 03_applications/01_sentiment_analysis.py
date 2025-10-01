"""
感情分析アプリケーション

このスクリプトでは、LLMを使った感情分析の実装例を示します。
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentAnalyzer:
    """感情分析用のクラス"""
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """感情分析モデルの初期化"""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # ラベルマッピング
        self.label_mapping = {
            0: "Negative",
            1: "Neutral", 
            2: "Positive"
        }
    
    def predict_sentiment(self, text):
        """単一テキストの感情分析"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'sentiment': self.label_mapping[predicted_class],
            'confidence': confidence,
            'all_scores': {
                self.label_mapping[i]: predictions[0][i].item() 
                for i in range(len(self.label_mapping))
            }
        }
    
    def predict_batch(self, texts):
        """複数テキストの感情分析"""
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            results.append(result)
        return results
    
    def evaluate(self, texts, true_labels):
        """モデルの評価"""
        predictions = self.predict_batch(texts)
        predicted_labels = [p['predicted_class'] for p in predictions]
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        report = classification_report(true_labels, predicted_labels, 
                                    target_names=list(self.label_mapping.values()))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': predictions
        }

def create_sample_data():
    """サンプルデータの作成"""
    sample_data = [
        {"text": "I love this product! It's amazing!", "label": 2},
        {"text": "This is terrible. I hate it.", "label": 0},
        {"text": "The weather is okay today.", "label": 1},
        {"text": "Fantastic! Best experience ever!", "label": 2},
        {"text": "I'm so disappointed with this service.", "label": 0},
        {"text": "The movie was average, nothing special.", "label": 1},
        {"text": "Outstanding quality and great value!", "label": 2},
        {"text": "Worst purchase I've ever made.", "label": 0},
        {"text": "It's fine, I guess.", "label": 1},
        {"text": "Absolutely brilliant! Highly recommend!", "label": 2},
        {"text": "Complete waste of money.", "label": 0},
        {"text": "Not bad, but could be better.", "label": 1},
        {"text": "Incredible! Exceeded all expectations!", "label": 2},
        {"text": "I regret buying this.", "label": 0},
        {"text": "It's decent, nothing more.", "label": 1}
    ]
    
    df = pd.DataFrame(sample_data)
    return df

def visualize_results(df, predictions):
    """結果の可視化"""
    # 予測結果をデータフレームに追加
    df['predicted_sentiment'] = [p['sentiment'] for p in predictions]
    df['confidence'] = [p['confidence'] for p in predictions]
    
    # 1. 感情分布の比較
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 実際の感情分布
    true_counts = df['label'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'}).value_counts()
    axes[0].pie(true_counts.values, labels=true_counts.index, autopct='%1.1f%%')
    axes[0].set_title('実際の感情分布')
    
    # 予測された感情分布
    pred_counts = df['predicted_sentiment'].value_counts()
    axes[1].pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
    axes[1].set_title('予測された感情分布')
    
    plt.tight_layout()
    plt.show()
    
    # 2. 信頼度の分布
    plt.figure(figsize=(10, 6))
    plt.hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('信頼度')
    plt.ylabel('頻度')
    plt.title('予測の信頼度分布')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 3. 混同行列
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df['label'], [p['predicted_class'] for p in predictions])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('予測')
    plt.ylabel('実際')
    plt.title('混同行列')
    plt.show()

def main():
    print("=== LLM感情分析アプリケーション ===\n")
    
    # 1. 感情分析器の初期化
    print("1. モデルの初期化")
    analyzer = SentimentAnalyzer()
    print(f"使用モデル: {analyzer.model_name}")
    print(f"対応ラベル: {list(analyzer.label_mapping.values())}\n")
    
    # 2. サンプルデータの準備
    print("2. サンプルデータの準備")
    df = create_sample_data()
    print(f"データ数: {len(df)}")
    print("\nサンプルデータ:")
    print(df.head())
    print()
    
    # 3. 単一テキストの感情分析
    print("3. 単一テキストの感情分析")
    test_text = "I absolutely love this new feature!"
    result = analyzer.predict_sentiment(test_text)
    print(f"テキスト: {test_text}")
    print(f"予測感情: {result['sentiment']}")
    print(f"信頼度: {result['confidence']:.3f}")
    print(f"全スコア: {result['all_scores']}")
    print()
    
    # 4. バッチ感情分析
    print("4. バッチ感情分析")
    texts = df['text'].tolist()
    true_labels = df['label'].tolist()
    
    predictions = analyzer.predict_batch(texts)
    print(f"分析完了: {len(predictions)}件")
    print()
    
    # 5. モデル評価
    print("5. モデル評価")
    evaluation = analyzer.evaluate(texts, true_labels)
    print(f"精度: {evaluation['accuracy']:.3f}")
    print("\n分類レポート:")
    print(evaluation['classification_report'])
    
    # 6. 詳細結果の表示
    print("\n6. 詳細結果")
    results_df = df.copy()
    results_df['predicted_sentiment'] = [p['sentiment'] for p in predictions]
    results_df['confidence'] = [p['confidence'] for p in predictions]
    results_df['correct'] = results_df['label'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'}) == results_df['predicted_sentiment']
    
    print("\n予測結果（最初の10件）:")
    print(results_df[['text', 'predicted_sentiment', 'confidence', 'correct']].head(10))
    
    # 7. 可視化
    print("\n7. 結果の可視化")
    try:
        visualize_results(df, predictions)
    except ImportError:
        print("可視化ライブラリが不足しています。matplotlibとseabornをインストールしてください。")
    
    # 8. 実用的な使用例
    print("\n8. 実用的な使用例")
    print("=== リアルタイム感情分析 ===")
    
    while True:
        user_input = input("\n分析したいテキストを入力してください（終了するには 'quit'）: ")
        if user_input.lower() == 'quit':
            break
        
        result = analyzer.predict_sentiment(user_input)
        print(f"感情: {result['sentiment']}")
        print(f"信頼度: {result['confidence']:.3f}")
        print(f"詳細スコア:")
        for sentiment, score in result['all_scores'].items():
            print(f"  {sentiment}: {score:.3f}")

if __name__ == "__main__":
    main()
