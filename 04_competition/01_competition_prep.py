"""
LLMコンペティション準備

このスクリプトでは、LLMコンペティションでよく使われるテクニックを実装します。
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import json
import os
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class CompetitionDataset(Dataset):
    """コンペティション用データセット"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EnsembleModel:
    """アンサンブルモデル"""
    
    def __init__(self, models, tokenizers, weights=None):
        self.models = models
        self.tokenizers = tokenizers
        self.weights = weights or [1.0] * len(models)
        self.num_models = len(models)
    
    def predict(self, text, max_length=100, temperature=0.7):
        """アンサンブル予測"""
        predictions = []
        
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            inputs = tokenizer.encode(text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(generated_text)
        
        # 重み付き平均（この例では単純化）
        return predictions

class DataAugmentation:
    """データ拡張クラス"""
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def back_translation(self, text, target_language="ja"):
        """バックトランスレーション"""
        # 簡略化された例（実際には翻訳APIが必要）
        # ここでは疑似実装
        return f"[Translated to {target_language}] {text} [Translated back]"
    
    def synonym_replacement(self, text, replacement_ratio=0.1):
        """同義語置換"""
        # 簡略化された例
        words = text.split()
        num_replacements = int(len(words) * replacement_ratio)
        
        # 簡単な同義語辞書
        synonyms = {
            "good": ["great", "excellent", "wonderful"],
            "bad": ["terrible", "awful", "horrible"],
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "miniature"]
        }
        
        for i in range(min(num_replacements, len(words))):
            word = words[i].lower().strip(".,!?")
            if word in synonyms:
                words[i] = np.random.choice(synonyms[word])
        
        return " ".join(words)
    
    def paraphrase_generation(self, text, num_paraphrases=3):
        """言い換え生成"""
        paraphrases = []
        for _ in range(num_paraphrases):
            # 簡略化された言い換え（実際にはより高度な手法が必要）
            paraphrased = text.replace("is", "can be").replace("are", "can be")
            paraphrases.append(paraphrased)
        return paraphrases

class CrossValidation:
    """クロスバリデーション"""
    
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    def evaluate_model(self, model, tokenizer, texts, labels):
        """モデル評価"""
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(texts, labels)):
            print(f"Fold {fold + 1}/{self.n_folds}")
            
            # データ分割
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # データセット作成
            train_dataset = CompetitionDataset(train_texts, train_labels, tokenizer)
            val_dataset = CompetitionDataset(val_texts, val_labels, tokenizer)
            
            # ここで実際の学習を行う（簡略化）
            # 実際の実装では、Trainerやカスタム学習ループを使用
            
            # 疑似評価スコア
            score = np.random.random()  # 実際の評価に置き換え
            fold_scores.append(score)
        
        return np.mean(fold_scores), np.std(fold_scores)

class PromptOptimization:
    """プロンプト最適化"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_prompt_variations(self, base_prompt, task_type="classification"):
        """プロンプトバリエーション生成"""
        variations = []
        
        if task_type == "classification":
            variations = [
                f"Classify the following text: {base_prompt}",
                f"Determine the category of: {base_prompt}",
                f"Label this text: {base_prompt}",
                f"Category: {base_prompt}",
                f"Class: {base_prompt}"
            ]
        elif task_type == "generation":
            variations = [
                f"Complete this: {base_prompt}",
                f"Continue: {base_prompt}",
                f"Write about: {base_prompt}",
                f"Generate text about: {base_prompt}",
                f"Create content: {base_prompt}"
            ]
        
        return variations
    
    def evaluate_prompts(self, prompts, test_data):
        """プロンプト評価"""
        prompt_scores = {}
        
        for prompt in prompts:
            scores = []
            for text, label in test_data:
                # プロンプトとテキストを結合
                full_prompt = f"{prompt}\n{text}"
                
                # 生成実行
                result = self.generate_text(full_prompt)
                
                # 評価（簡略化）
                score = self.calculate_score(result, label)
                scores.append(score)
            
            prompt_scores[prompt] = np.mean(scores)
        
        return prompt_scores
    
    def generate_text(self, prompt, max_length=100):
        """テキスト生成"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def calculate_score(self, generated, target):
        """スコア計算（簡略化）"""
        # 実際の実装では、タスクに応じた適切な評価指標を使用
        return np.random.random()

class CompetitionPipeline:
    """コンペティション用パイプライン"""
    
    def __init__(self):
        self.models = []
        self.tokenizers = []
        self.ensemble_model = None
        self.data_augmentation = None
        self.cross_validation = CrossValidation()
        self.prompt_optimization = None
    
    def add_model(self, model_name):
        """モデル追加"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.models.append(model)
        self.tokenizers.append(tokenizer)
        
        if self.prompt_optimization is None:
            self.prompt_optimization = PromptOptimization(model, tokenizer)
    
    def setup_ensemble(self, weights=None):
        """アンサンブル設定"""
        if len(self.models) < 2:
            print("アンサンブルには2つ以上のモデルが必要です")
            return
        
        self.ensemble_model = EnsembleModel(
            self.models, 
            self.tokenizers, 
            weights
        )
    
    def optimize_prompts(self, base_prompts, test_data, task_type="classification"):
        """プロンプト最適化"""
        all_variations = []
        
        for base_prompt in base_prompts:
            variations = self.prompt_optimization.generate_prompt_variations(
                base_prompt, task_type
            )
            all_variations.extend(variations)
        
        # プロンプト評価
        prompt_scores = self.prompt_optimization.evaluate_prompts(
            all_variations, test_data
        )
        
        # 最良のプロンプトを選択
        best_prompts = sorted(
            prompt_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return best_prompts
    
    def cross_validate(self, texts, labels):
        """クロスバリデーション実行"""
        if not self.models:
            print("モデルが設定されていません")
            return None
        
        # 最初のモデルでクロスバリデーション
        mean_score, std_score = self.cross_validation.evaluate_model(
            self.models[0], 
            self.tokenizers[0], 
            texts, 
            labels
        )
        
        return mean_score, std_score

def create_competition_data():
    """コンペティションデータの作成"""
    # サンプルデータ
    data = {
        'text': [
            "This product is amazing! I love it!",
            "Terrible quality, very disappointed.",
            "It's okay, nothing special.",
            "Outstanding service and great value!",
            "Worst purchase ever, complete waste.",
            "Good product, would recommend.",
            "Excellent quality and fast delivery!",
            "Poor customer service, very rude.",
            "Average product, meets expectations.",
            "Fantastic! Exceeded all my expectations!"
        ],
        'label': [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]  # 0: negative, 1: positive
    }
    
    return pd.DataFrame(data)

def main():
    print("=== LLMコンペティション準備 ===\n")
    
    # 1. パイプラインの初期化
    print("1. コンペティションパイプラインの初期化")
    pipeline = CompetitionPipeline()
    
    # 2. 複数モデルの追加
    print("2. 複数モデルの追加")
    model_names = ["gpt2", "gpt2-medium"]  # 実際にはより多様なモデルを使用
    
    for model_name in model_names:
        try:
            print(f"  {model_name} を追加中...")
            pipeline.add_model(model_name)
            print(f"  {model_name} 追加完了")
        except Exception as e:
            print(f"  {model_name} の追加に失敗: {e}")
    
    print(f"追加されたモデル数: {len(pipeline.models)}\n")
    
    # 3. データの準備
    print("3. データの準備")
    df = create_competition_data()
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    print(f"データ数: {len(df)}")
    print(f"クラス分布: {df['label'].value_counts().to_dict()}\n")
    
    # 4. アンサンブル設定
    print("4. アンサンブル設定")
    if len(pipeline.models) >= 2:
        pipeline.setup_ensemble(weights=[0.6, 0.4])
        print("アンサンブルモデルを設定しました")
    else:
        print("アンサンブルには2つ以上のモデルが必要です")
    print()
    
    # 5. プロンプト最適化
    print("5. プロンプト最適化")
    base_prompts = [
        "Classify the sentiment of this text:",
        "Determine if this is positive or negative:",
        "Label the emotion in this text:"
    ]
    
    test_data = list(zip(texts[:5], labels[:5]))  # サンプルデータ
    
    try:
        best_prompts = pipeline.optimize_prompts(
            base_prompts, test_data, "classification"
        )
        print("最良のプロンプト:")
        for prompt, score in best_prompts:
            print(f"  スコア: {score:.3f} - {prompt}")
    except Exception as e:
        print(f"プロンプト最適化でエラー: {e}")
    print()
    
    # 6. クロスバリデーション
    print("6. クロスバリデーション")
    try:
        mean_score, std_score = pipeline.cross_validate(texts, labels)
        print(f"平均スコア: {mean_score:.3f} ± {std_score:.3f}")
    except Exception as e:
        print(f"クロスバリデーションでエラー: {e}")
    print()
    
    # 7. 実用的なヒント
    print("7. コンペティション成功のヒント")
    tips = [
        "データの前処理とクリーニングを徹底する",
        "複数のモデルをアンサンブルする",
        "クロスバリデーションで過学習を防ぐ",
        "プロンプトエンジニアリングを最適化する",
        "データ拡張で学習データを増やす",
        "ハイパーパラメータを系統的に調整する",
        "アブレーションスタディで要因を特定する",
        "最終提出前に複数回テストする"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")
    
    print("\n=== コンペティション準備完了 ===")

if __name__ == "__main__":
    main()
