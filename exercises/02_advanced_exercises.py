"""
LLM高度演習問題

このファイルには、LLMの高度な機能を学習するための演習問題が含まれています。
実践的なアプリケーション開発に必要なスキルを習得できます。
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

class AdvancedLLMExerciseSolver:
    """高度なLLM演習問題解決クラス"""
    
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def exercise_1_sentiment_analysis_app(self):
        """演習1: 感情分析アプリケーション"""
        print("=== 演習1: 感情分析アプリケーション ===")
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
    
    def exercise_2_text_summarization_app(self):
        """演習2: テキスト要約アプリケーション"""
        print("\n=== 演習2: テキスト要約アプリケーション ===")
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
    
    def exercise_3_question_answering_system(self):
        """演習3: 質問応答システム"""
        print("\n=== 演習3: 質問応答システム ===")
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
        """演習4: チャットボット開発"""
        print("\n=== 演習4: チャットボット開発 ===")
        print("対話型チャットボットを実装してください。")
        
        print("実装すべき機能:")
        print("- 会話履歴の管理")
        print("- コンテキストの保持")
        print("- 適切な応答生成")
        print("- エラーハンドリング")
        
        # ここにチャットボットを実装してください
        # ヒント: 会話履歴をプロンプトに含める
    
    def exercise_5_model_fine_tuning(self):
        """演習5: モデルのファインチューニング"""
        print("\n=== 演習5: モデルのファインチューニング ===")
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
    
    def exercise_6_performance_evaluation(self):
        """演習6: パフォーマンス評価"""
        print("\n=== 演習6: パフォーマンス評価 ===")
        print("モデルの性能を評価してください。")
        
        print("実装すべき評価指標:")
        print("- 精度（Accuracy）")
        print("- F1スコア")
        print("- 混同行列")
        print("- 分類レポート")
        print("- 可視化")
        
        # ここにパフォーマンス評価を実装してください
        # ヒント: sklearn.metricsを使用
    
    def exercise_7_hyperparameter_optimization(self):
        """演習7: ハイパーパラメータ最適化"""
        print("\n=== 演習7: ハイパーパラメータ最適化 ===")
        print("ハイパーパラメータを最適化してください。")
        
        print("最適化すべきパラメータ:")
        print("- 学習率（learning_rate）")
        print("- バッチサイズ（batch_size）")
        print("- エポック数（num_epochs）")
        print("- 温度（temperature）")
        print("- 重み減衰（weight_decay）")
        
        # ここにハイパーパラメータ最適化を実装してください
        # ヒント: グリッドサーチまたはランダムサーチを使用
    
    def exercise_8_model_ensemble(self):
        """演習8: モデルアンサンブル"""
        print("\n=== 演習8: モデルアンサンブル ===")
        print("複数のモデルをアンサンブルしてください。")
        
        print("実装すべきアンサンブル手法:")
        print("- 重み付き平均")
        print("- 投票法")
        print("- スタッキング")
        print("- ブレンディング")
        
        # ここにモデルアンサンブルを実装してください
        # ヒント: 複数のモデルの予測を組み合わせる
    
    def exercise_9_deployment_preparation(self):
        """演習9: デプロイメント準備"""
        print("\n=== 演習9: デプロイメント準備 ===")
        print("本番環境でのデプロイメントを準備してください。")
        
        print("実装すべき機能:")
        print("- モデルの保存と読み込み")
        print("- APIエンドポイントの作成")
        print("- エラーハンドリング")
        print("- ログ記録")
        print("- パフォーマンス監視")
        
        # ここにデプロイメント準備を実装してください
        # ヒント: FlaskやFastAPIを使用
    
    def exercise_10_advanced_techniques(self):
        """演習10: 高度なテクニック"""
        print("\n=== 演習10: 高度なテクニック ===")
        print("高度なテクニックを実装してください。")
        
        print("実装すべきテクニック:")
        print("- プロンプトチェーン")
        print("- 条件付き生成")
        print("- 制約付き生成")
        print("- 多段階推論")
        print("- 自己修正生成")
        
        # ここに高度なテクニックを実装してください
    
    def run_all_exercises(self):
        """すべての演習を実行"""
        print("🚀 LLM高度演習を開始します")
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
        self.exercise_10_advanced_techniques()
        
        print("\n✅ すべての演習が完了しました！")
        print("各演習の解答を実装して、実践的なスキルを身につけてください。")

def main():
    """メイン関数"""
    print("LLM高度演習問題")
    print("=" * 30)
    
    # 演習ソルバーの初期化
    solver = AdvancedLLMExerciseSolver()
    
    # 演習の実行
    solver.run_all_exercises()

if __name__ == "__main__":
    main()
