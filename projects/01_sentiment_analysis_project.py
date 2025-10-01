"""
感情分析プロジェクト

このプロジェクトでは、実用的な感情分析アプリケーションを開発します。
実際のビジネスで使用できるレベルの品質を目指します。
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json
import os
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalysisProject:
    """感情分析プロジェクトクラス"""
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.results = {}
    
    def setup_model(self):
        """モデルのセットアップ"""
        print("🔄 モデルをセットアップ中...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            print(f"✅ モデル '{self.model_name}' の読み込み完了")
            print(f"   パラメータ数: {self.model.num_parameters():,}")
            print(f"   語彙数: {len(self.tokenizer)}")
            
            return True
            
        except Exception as e:
            print(f"❌ モデルの読み込みに失敗: {e}")
            return False
    
    def create_sample_dataset(self):
        """サンプルデータセットの作成"""
        print("📊 サンプルデータセットを作成中...")
        
        # 実際のビジネスデータを模擬したサンプル
        data = {
            'text': [
                # ポジティブなレビュー
                "I absolutely love this product! It exceeded my expectations and I would definitely buy it again.",
                "Amazing quality and fast delivery. Highly recommend to everyone!",
                "Outstanding customer service and excellent product quality.",
                "This is the best purchase I've made this year. Worth every penny!",
                "Fantastic experience from start to finish. 5 stars!",
                
                # ネガティブなレビュー
                "Terrible product. Complete waste of money and time.",
                "Poor quality and worse customer service. Would not recommend.",
                "Disappointed with this purchase. Not as described at all.",
                "Worst experience ever. Avoid this product at all costs.",
                "Regret buying this. Money down the drain.",
                
                # ニュートラルなレビュー
                "The product is okay. Nothing special but does the job.",
                "Average quality. Expected more for the price.",
                "It's fine, I guess. Not great but not terrible either.",
                "The product works as expected. Nothing more, nothing less.",
                "Decent product but could be better. Middle of the road.",
                
                # より多様なサンプル
                "Great value for money! The quality is surprisingly good for the price.",
                "Not impressed with the build quality. Feels cheap and flimsy.",
                "The product arrived on time and works well. Satisfied with the purchase.",
                "Overpriced for what you get. There are better alternatives available.",
                "Excellent packaging and presentation. The product itself is decent.",
                
                # 複雑な感情表現
                "I have mixed feelings about this product. The design is great but the functionality is lacking.",
                "Love the concept but the execution could be better. Still worth trying.",
                "The product has potential but needs improvement. Not ready for prime time.",
                "Good idea, poor implementation. Hope they fix the issues in the next version.",
                "Interesting product with some innovative features, but not perfect yet."
            ],
            'label': [2, 2, 2, 2, 2,  # ポジティブ
                     0, 0, 0, 0, 0,  # ネガティブ
                     1, 1, 1, 1, 1,  # ニュートラル
                     2, 0, 2, 0, 1,  # 追加サンプル
                     1, 2, 1, 0, 1]  # 複雑な感情
        }
        
        self.df = pd.DataFrame(data)
        print(f"✅ データセット作成完了: {len(self.df)}件")
        print(f"   ポジティブ: {sum(self.df['label'] == 2)}件")
        print(f"   ネガティブ: {sum(self.df['label'] == 0)}件")
        print(f"   ニュートラル: {sum(self.df['label'] == 1)}件")
        
        return self.df
    
    def analyze_dataset(self):
        """データセットの分析"""
        print("\\n📈 データセットの分析")
        print("=" * 40)
        
        # 基本統計
        print(f"総データ数: {len(self.df)}")
        print(f"平均文字数: {self.df['text'].str.len().mean():.1f}")
        print(f"平均単語数: {self.df['text'].str.split().str.len().mean():.1f}")
        
        # ラベル分布
        label_counts = self.df['label'].value_counts().sort_index()
        print("\\nラベル分布:")
        for label, count in label_counts.items():
            sentiment = self.label_mapping[label]
            percentage = count / len(self.df) * 100
            print(f"  {sentiment}: {count}件 ({percentage:.1f}%)")
        
        # 可視化
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # ラベル分布の円グラフ
        axes[0].pie(label_counts.values, labels=[self.label_mapping[i] for i in label_counts.index], 
                   autopct='%1.1f%%', startangle=90)
        axes[0].set_title('感情分布')
        
        # テキスト長の分布
        text_lengths = self.df['text'].str.len()
        axes[1].hist(text_lengths, bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('文字数')
        axes[1].set_ylabel('頻度')
        axes[1].set_title('テキスト長の分布')
        
        plt.tight_layout()
        plt.show()
    
    def predict_sentiment(self, text):
        """単一テキストの感情分析"""
        if self.model is None or self.tokenizer is None:
            return {"error": "モデルが読み込まれていません"}
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            'text': text,
            'predicted_class': predicted_class,
            'sentiment': self.label_mapping[predicted_class],
            'confidence': confidence,
            'all_scores': {
                self.label_mapping[i]: predictions[0][i].item() 
                for i in range(len(self.label_mapping))
            }
        }
    
    def evaluate_model(self, test_data=None):
        """モデルの評価"""
        print("\\n📊 モデルの評価")
        print("=" * 40)
        
        if test_data is None:
            test_data = self.df
        
        predictions = []
        confidences = []
        
        for text in test_data['text']:
            result = self.predict_sentiment(text)
            predictions.append(result['predicted_class'])
            confidences.append(result['confidence'])
        
        # 評価指標の計算
        accuracy = accuracy_score(test_data['label'], predictions)
        f1 = f1_score(test_data['label'], predictions, average='weighted')
        
        print(f"精度: {accuracy:.3f}")
        print(f"F1スコア: {f1:.3f}")
        
        # 分類レポート
        print("\\n分類レポート:")
        report = classification_report(
            test_data['label'], 
            predictions, 
            target_names=list(self.label_mapping.values())
        )
        print(report)
        
        # 混同行列
        cm = confusion_matrix(test_data['label'], predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=list(self.label_mapping.values()),
                   yticklabels=list(self.label_mapping.values()))
        plt.xlabel('予測')
        plt.ylabel('実際')
        plt.title('混同行列')
        plt.show()
        
        # 信頼度の分布
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('信頼度')
        plt.ylabel('頻度')
        plt.title('予測の信頼度分布')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'平均: {np.mean(confidences):.3f}')
        plt.legend()
        plt.show()
        
        self.results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': predictions,
            'confidences': confidences
        }
        
        return self.results
    
    def interactive_demo(self):
        """インタラクティブデモ"""
        print("\\n🎮 インタラクティブデモ")
        print("=" * 40)
        print("感情分析を試してみてください！（終了するには 'quit' と入力）")
        
        while True:
            try:
                text = input("\\nテキストを入力してください: ")
                
                if text.lower() == 'quit':
                    print("デモを終了します。")
                    break
                
                if not text.strip():
                    print("テキストを入力してください。")
                    continue
                
                result = self.predict_sentiment(text)
                
                print(f"\\n📊 分析結果:")
                print(f"テキスト: {result['text']}")
                print(f"感情: {result['sentiment']}")
                print(f"信頼度: {result['confidence']:.3f}")
                print(f"詳細スコア:")
                for sentiment, score in result['all_scores'].items():
                    print(f"  {sentiment}: {score:.3f}")
                
            except KeyboardInterrupt:
                print("\\nデモを終了します。")
                break
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                continue
    
    def batch_analysis(self, texts):
        """バッチ分析"""
        print(f"\\n📦 バッチ分析: {len(texts)}件")
        print("=" * 40)
        
        results = []
        for i, text in enumerate(texts, 1):
            result = self.predict_sentiment(text)
            results.append(result)
            print(f"{i:2d}. {text[:50]}... -> {result['sentiment']} ({result['confidence']:.3f})")
        
        return results
    
    def export_results(self, filename="sentiment_analysis_results.json"):
        """結果のエクスポート"""
        print(f"\\n💾 結果をエクスポート中: {filename}")
        
        export_data = {
            'model_name': self.model_name,
            'label_mapping': self.label_mapping,
            'results': self.results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 結果を {filename} に保存しました")
    
    def run_complete_project(self):
        """完全なプロジェクトの実行"""
        print("🚀 感情分析プロジェクトを開始します")
        print("=" * 50)
        
        # 1. モデルのセットアップ
        if not self.setup_model():
            print("❌ プロジェクトを終了します")
            return
        
        # 2. データセットの作成
        self.create_sample_dataset()
        
        # 3. データセットの分析
        self.analyze_dataset()
        
        # 4. モデルの評価
        self.evaluate_model()
        
        # 5. インタラクティブデモ
        self.interactive_demo()
        
        # 6. 結果のエクスポート
        self.export_results()
        
        print("\\n✅ プロジェクトが完了しました！")
        print("感情分析アプリケーションが正常に動作しています。")

def main():
    """メイン関数"""
    print("感情分析プロジェクト")
    print("=" * 30)
    
    # プロジェクトの初期化
    project = SentimentAnalysisProject()
    
    # 完全なプロジェクトの実行
    project.run_complete_project()

if __name__ == "__main__":
    main()
