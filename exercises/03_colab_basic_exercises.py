"""
LLM基礎演習問題（Google Colab版）

このファイルには、Google Colab環境でLLMの基礎を学習するための演習問題が含まれています。
Colab環境に最適化された実装になっています。
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class ColabLLMExerciseSolver:
    """Colab用LLM演習問題解決クラス"""
    
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def exercise_1_basic_generation(self):
        """演習1: 基本的なテキスト生成（Colab最適化）"""
        print("=== 演習1: 基本的なテキスト生成（Colab版） ===")
        print("以下のプロンプトからテキストを生成してください。")
        
        prompts = [
            "The future of artificial intelligence is",
            "Once upon a time, there was a",
            "In the world of machine learning,"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- プロンプト {i} ---")
            print(f"入力: {prompt}")
            
            # ここに生成コードを書いてください
            # ヒント: generate_text関数を使用
            
            # 解答例（コメントアウト）
            # generated = self.generate_text(prompt, max_length=80)
            # print(f"出力: {generated}")
    
    def exercise_2_parameter_tuning(self):
        """演習2: パラメータの調整（Colab最適化）"""
        print("\n=== 演習2: パラメータの調整（Colab版） ===")
        print("同じプロンプトで異なる温度パラメータを使用して生成してください。")
        
        prompt = "The secret to success is"
        temperatures = [0.3, 0.7, 1.0]
        
        print(f"プロンプト: {prompt}")
        
        for temp in temperatures:
            print(f"\n温度: {temp}")
            # ここに生成コードを書いてください
            # ヒント: temperatureパラメータを変更
    
    def exercise_3_prompt_engineering(self):
        """演習3: プロンプトエンジニアリング（Colab最適化）"""
        print("\n=== 演習3: プロンプトエンジニアリング（Colab版） ===")
        print("同じタスクに対して異なるプロンプトスタイルを試してください。")
        
        task = "感情分析"
        text = "I absolutely love this new product!"
        
        prompt_styles = [
            f"Classify the sentiment: {text}",
            f"Determine if this is positive or negative: {text}",
            f"Analyze the emotion in this text: {text}",
            f"Rate the sentiment (positive/negative/neutral): {text}"
        ]
        
        for i, prompt in enumerate(prompt_styles, 1):
            print(f"\n--- スタイル {i} ---")
            print(f"プロンプト: {prompt}")
            # ここに生成コードを書いてください
    
    def exercise_4_few_shot_learning(self):
        """演習4: Few-shot学習（Colab最適化）"""
        print("\n=== 演習4: Few-shot学習（Colab版） ===")
        print("例示を含むプロンプトを作成してください。")
        
        examples = [
            ("I love this movie!", "positive"),
            ("This is terrible.", "negative"),
            ("It's okay.", "neutral")
        ]
        
        query = "This is amazing!"
        
        # ここにFew-shotプロンプトを作成してください
        # ヒント: 例示をプロンプトに含める
    
    def exercise_5_text_analysis(self):
        """演習5: テキスト分析（Colab最適化）"""
        print("\n=== 演習5: テキスト分析（Colab版） ===")
        print("生成されたテキストの特徴を分析してください。")
        
        prompt = "Write a short story about a robot"
        generated_text = self.generate_text(prompt, max_length=150)
        
        print(f"生成されたテキスト: {generated_text}")
        
        # ここに分析コードを書いてください
        # ヒント: 文字数、単語数、文数などを計算
    
    def exercise_6_colab_specific(self):
        """演習6: Colab固有の機能"""
        print("\n=== 演習6: Colab固有の機能 ===")
        print("Colab環境での特殊な機能を試してください。")
        
        # 1. GPU使用状況の確認
        print("1. GPU使用状況:")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   メモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        else:
            print("   GPUが利用できません")
        
        # 2. メモリ効率の確認
        print("\n2. メモリ効率:")
        print(f"   モデルパラメータ数: {self.model.num_parameters():,}")
        print(f"   語彙数: {len(self.tokenizer)}")
        
        # 3. 生成速度の測定
        print("\n3. 生成速度測定:")
        import time
        start_time = time.time()
        generated = self.generate_text("Test prompt", max_length=50)
        end_time = time.time()
        print(f"   生成時間: {end_time - start_time:.2f}秒")
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, **kwargs):
        """テキスト生成のヘルパー関数（Colab最適化）"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def run_all_exercises(self):
        """すべての演習を実行（Colab最適化）"""
        print("🚀 LLM基礎演習を開始します（Colab版）")
        print("=" * 50)
        
        self.exercise_1_basic_generation()
        self.exercise_2_parameter_tuning()
        self.exercise_3_prompt_engineering()
        self.exercise_4_few_shot_learning()
        self.exercise_5_text_analysis()
        self.exercise_6_colab_specific()
        
        print("\n✅ すべての演習が完了しました！")
        print("各演習の解答を確認して、理解を深めてください。")
        print("\n💡 Colab環境でのアドバイス:")
        print("• 定期的にランタイムを再起動してメモリをクリア")
        print("• 大きなモデルは段階的に読み込み")
        print("• 結果は定期的にGoogle Driveに保存")

def main():
    """メイン関数"""
    print("LLM基礎演習問題（Google Colab版）")
    print("=" * 40)
    
    # 演習ソルバーの初期化
    solver = ColabLLMExerciseSolver()
    
    # 演習の実行
    solver.run_all_exercises()

if __name__ == "__main__":
    main()
