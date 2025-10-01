"""
LLM高度演習問題（Google Colab版）

このファイルには、Google Colab環境でLLMの高度な機能を学習するための演習問題が含まれています。
Colab環境に最適化された実装になっています。
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import json
import os
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class ColabAdvancedLLMExerciseSolver:
    """Colab用高度なLLM演習問題解決クラス"""
    
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def exercise_1_sentiment_analysis_app(self):
        """演習1: 感情分析アプリケーション（Colab最適化）"""
        print("=== 演習1: 感情分析アプリケーション（Colab版） ===")
        print("感情分析アプリケーションを実装してください。")
        
        # サンプルデータ
        sample_texts = [
            "I love this product! It's amazing!",
            "This is terrible. I hate it.",
            "The weather is okay today.",
            "Fantastic! Best experience ever!",
            "I'm so disappointed with this service."
        ]
        
        print("サンプルテキスト:")
        for i, text in enumerate(sample_texts, 1):
            print(f"{i}. {text}")
        
        # ここに感情分析アプリケーションを実装してください
        # ヒント: プロンプトエンジニアリングとFew-shot学習を使用
        
        print("\n実装すべき機能:")
        print("- テキストの感情を分類（positive/negative/neutral）")
        print("- 信頼度スコアの計算")
        print("- バッチ処理の対応")
        print("- 結果の可視化")
        print("- Colab環境でのメモリ効率化")
    
    def exercise_2_text_summarization_app(self):
        """演習2: テキスト要約アプリケーション（Colab最適化）"""
        print("\n=== 演習2: テキスト要約アプリケーション（Colab版） ===")
        print("テキスト要約アプリケーションを実装してください。")
        
        # サンプルテキスト
        long_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. The term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
        """
        
        print(f"元のテキスト: {long_text.strip()}")
        
        # ここにテキスト要約アプリケーションを実装してください
        # ヒント: プロンプトテンプレートとパラメータ調整を使用
        
        print("\n実装すべき機能:")
        print("- 長いテキストの要約")
        print("- 要約の長さ調整")
        print("- キーポイントの抽出")
        print("- 要約品質の評価")
        print("- Colab環境でのメモリ効率化")
    
    def exercise_3_question_answering_system(self):
        """演習3: 質問応答システム（Colab最適化）"""
        print("\n=== 演習3: 質問応答システム（Colab版） ===")
        print("質問応答システムを実装してください。")
        
        # サンプルデータ
        context = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning. Deep learning is a subset of machine learning that uses neural networks with multiple layers.
        """
        
        questions = [
            "What is machine learning?",
            "What are the types of machine learning?",
            "How is deep learning related to machine learning?"
        ]
        
        print(f"文脈: {context.strip()}")
        print("\n質問:")
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
        
        # ここに質問応答システムを実装してください
        # ヒント: 文脈と質問を組み合わせたプロンプトを作成
    
    def exercise_4_chatbot_development(self):
        """演習4: チャットボット開発（Colab最適化）"""
        print("\n=== 演習4: チャットボット開発（Colab版） ===")
        print("対話型チャットボットを実装してください。")
        
        print("実装すべき機能:")
        print("- 会話履歴の管理")
        print("- コンテキストの保持")
        print("- 適切な応答生成")
        print("- エラーハンドリング")
        print("- Colab環境でのメモリ効率化")
        
        # ここにチャットボットを実装してください
        # ヒント: 会話履歴をプロンプトに含める
    
    def exercise_5_model_fine_tuning(self):
        """演習5: モデルのファインチューニング（Colab最適化）"""
        print("\n=== 演習5: モデルのファインチューニング（Colab版） ===")
        print("カスタムデータセットでモデルをファインチューニングしてください。")
        
        # サンプルデータセット
        sample_data = {
            'texts': [
                "This is a positive review about the product.",
                "I hate this terrible service.",
                "The quality is average, nothing special.",
                "Amazing! Highly recommend this item.",
                "Poor quality, would not buy again."
            ],
            'labels': [1, 0, 1, 1, 0]  # 0: negative, 1: positive
        }
        
        print("サンプルデータセット:")
        for text, label in zip(sample_data['texts'], sample_data['labels']):
            print(f"テキスト: {text} | ラベル: {label}")
        
        # ここにファインチューニングを実装してください
        # ヒント: TrainingArgumentsとTrainerクラスを使用
        # Colab環境では軽量化設定を推奨
    
    def exercise_6_performance_evaluation(self):
        """演習6: パフォーマンス評価（Colab最適化）"""
        print("\n=== 演習6: パフォーマンス評価（Colab版） ===")
        print("モデルの性能を評価してください。")
        
        print("実装すべき評価指標:")
        print("- 精度（Accuracy）")
        print("- F1スコア")
        print("- 混同行列")
        print("- 分類レポート")
        print("- 可視化")
        print("- Colab環境でのメモリ効率化")
        
        # ここにパフォーマンス評価を実装してください
        # ヒント: sklearn.metricsを使用
    
    def exercise_7_hyperparameter_optimization(self):
        """演習7: ハイパーパラメータ最適化（Colab最適化）"""
        print("\n=== 演習7: ハイパーパラメータ最適化（Colab版） ===")
        print("ハイパーパラメータを最適化してください。")
        
        print("最適化すべきパラメータ:")
        print("- 学習率（learning_rate）")
        print("- バッチサイズ（batch_size）")
        print("- エポック数（num_epochs）")
        print("- 温度（temperature）")
        print("- 重み減衰（weight_decay）")
        print("- Colab環境での制限を考慮")
        
        # ここにハイパーパラメータ最適化を実装してください
        # ヒント: グリッドサーチまたはランダムサーチを使用
        # Colab環境では試行回数を制限
    
    def exercise_8_model_ensemble(self):
        """演習8: モデルアンサンブル（Colab最適化）"""
        print("\n=== 演習8: モデルアンサンブル（Colab版） ===")
        print("複数のモデルをアンサンブルしてください。")
        
        print("実装すべきアンサンブル手法:")
        print("- 重み付き平均")
        print("- 投票法")
        print("- スタッキング")
        print("- ブレンディング")
        print("- Colab環境でのメモリ効率化")
        
        # ここにモデルアンサンブルを実装してください
        # ヒント: 複数のモデルの予測を組み合わせる
    
    def exercise_9_deployment_preparation(self):
        """演習9: デプロイメント準備（Colab最適化）"""
        print("\n=== 演習9: デプロイメント準備（Colab版） ===")
        print("本番環境でのデプロイメントを準備してください。")
        
        print("実装すべき機能:")
        print("- モデルの保存と読み込み")
        print("- APIエンドポイントの作成")
        print("- エラーハンドリング")
        print("- ログ記録")
        print("- パフォーマンス監視")
        print("- Colab環境での制限を考慮")
        
        # ここにデプロイメント準備を実装してください
        # ヒント: FlaskやFastAPIを使用
    
    def exercise_10_colab_specific_advanced(self):
        """演習10: Colab固有の高度な機能"""
        print("\n=== 演習10: Colab固有の高度な機能 ===")
        print("Colab環境での高度な機能を実装してください。")
        
        print("実装すべき機能:")
        print("- メモリ使用量の監視と最適化")
        print("- 段階的なモデル読み込み")
        print("- 結果のGoogle Drive保存")
        print("- ランタイム再起動の自動化")
        print("- エラー時の自動復旧")
        print("- 進捗の可視化")
        
        # ここにColab固有の高度な機能を実装してください
    
    def run_all_exercises(self):
        """すべての演習を実行（Colab最適化）"""
        print("🚀 LLM高度演習を開始します（Colab版）")
        print("=" * 50)
        
        self.exercise_1_sentiment_analysis_app()
        self.exercise_2_text_summarization_app()
        self.exercise_3_question_answering_system()
        self.exercise_4_chatbot_development()
        self.exercise_5_model_fine_tuning()
        self.exercise_6_performance_evaluation()
        self.exercise_7_hyperparameter_optimization()
        self.exercise_8_model_ensemble()
        self.exercise_9_deployment_preparation()
        self.exercise_10_colab_specific_advanced()
        
        print("\n✅ すべての演習が完了しました！")
        print("各演習の解答を実装して、実践的なスキルを身につけてください。")
        print("\n💡 Colab環境での実践的なアドバイス:")
        print("• 定期的にランタイムを再起動してメモリをクリア")
        print("• 大きなモデルは段階的に読み込み")
        print("• 結果は定期的にGoogle Driveに保存")
        print("• エラーが発生したら設定を軽量化")
        print("• 進捗は可視化して確認")

def main():
    """メイン関数"""
    print("LLM高度演習問題（Google Colab版）")
    print("=" * 40)
    
    # 演習ソルバーの初期化
    solver = ColabAdvancedLLMExerciseSolver()
    
    # 演習の実行
    solver.run_all_exercises()

if __name__ == "__main__":
    main()
