"""
LLMファインチューニングの基礎

このスクリプトでは、事前学習済みモデルのファインチューニングの基本を学びます。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import json
import os

class TextDataset(Dataset):
    """テキストデータセット用のカスタムクラス"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
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
            'labels': encoding['input_ids'].flatten()
        }

class FineTuner:
    """ファインチューニング用のクラス"""
    
    def __init__(self, model_name="gpt2", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # パディングトークンの設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # モデルの設定
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def prepare_data(self, texts):
        """データの準備"""
        dataset = TextDataset(texts, self.tokenizer, self.max_length)
        return dataset
    
    def fine_tune(self, dataset, output_dir="./fine_tuned_model", 
                  num_epochs=3, batch_size=4, learning_rate=5e-5):
        """ファインチューニングの実行"""
        
        # データコレーターの設定
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # GPT-2は因果言語モデルなのでFalse
        )
        
        # トレーニング引数の設定
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            save_total_limit=2,
            learning_rate=learning_rate,
            fp16=True,  # メモリ効率化
        )
        
        # トレーナーの初期化
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # ファインチューニングの実行
        print("ファインチューニングを開始します...")
        trainer.train()
        
        # モデルの保存
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        print(f"モデルが {output_dir} に保存されました。")
        
        return trainer
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """生成されたテキストのテスト"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

def create_sample_data():
    """サンプルデータの作成"""
    # 簡単な例：プログラミング関連のテキスト
    sample_texts = [
        "Python is a powerful programming language that is easy to learn and use.",
        "Machine learning algorithms can help solve complex problems in various domains.",
        "Data preprocessing is an important step in any machine learning pipeline.",
        "Deep learning models require large amounts of data to train effectively.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision algorithms can analyze and interpret visual information.",
        "Reinforcement learning agents learn through interaction with their environment.",
        "Feature engineering involves creating meaningful input variables for models.",
        "Model evaluation is crucial for assessing the performance of machine learning systems.",
        "Cross-validation helps ensure that models generalize well to unseen data."
    ]
    return sample_texts

def main():
    print("=== LLMファインチューニングの基礎 ===\n")
    
    # 1. ファインチューナーの初期化
    print("1. モデルの初期化")
    fine_tuner = FineTuner(model_name="gpt2", max_length=256)
    print(f"モデル: {fine_tuner.model_name}")
    print(f"パラメータ数: {fine_tuner.model.num_parameters():,}\n")
    
    # 2. サンプルデータの準備
    print("2. データの準備")
    sample_texts = create_sample_data()
    print(f"サンプルテキスト数: {len(sample_texts)}")
    for i, text in enumerate(sample_texts[:3], 1):
        print(f"  {i}. {text}")
    print("  ...\n")
    
    # 3. ファインチューニング前のテスト
    print("3. ファインチューニング前の生成テスト")
    test_prompt = "Machine learning is"
    original_result = fine_tuner.generate_text(test_prompt, max_length=50)
    print(f"プロンプト: {test_prompt}")
    print(f"生成結果: {original_result}\n")
    
    # 4. データセットの準備
    print("4. データセットの準備")
    dataset = fine_tuner.prepare_data(sample_texts)
    print(f"データセットサイズ: {len(dataset)}\n")
    
    # 5. ファインチューニングの実行（軽量版）
    print("5. ファインチューニングの実行")
    print("注意: 実際のファインチューニングには時間がかかります。")
    print("この例では軽量な設定で実行します。\n")
    
    # 軽量な設定でファインチューニング
    try:
        trainer = fine_tuner.fine_tune(
            dataset=dataset,
            output_dir="./fine_tuned_gpt2",
            num_epochs=1,  # 軽量化のため1エポックのみ
            batch_size=2,  # 軽量化のため小さなバッチサイズ
            learning_rate=5e-5
        )
        print("ファインチューニングが完了しました！\n")
        
        # 6. ファインチューニング後のテスト
        print("6. ファインチューニング後の生成テスト")
        fine_tuned_result = fine_tuner.generate_text(test_prompt, max_length=50)
        print(f"プロンプト: {test_prompt}")
        print(f"生成結果: {fine_tuned_result}\n")
        
        # 7. 比較
        print("7. ファインチューニング前後の比較")
        print(f"元のモデル: {original_result}")
        print(f"ファインチューニング後: {fine_tuned_result}")
        
    except Exception as e:
        print(f"ファインチューニング中にエラーが発生しました: {e}")
        print("これは通常、メモリ不足や設定の問題が原因です。")
        print("実際の環境では、より大きなGPUメモリや適切な設定が必要です。")

if __name__ == "__main__":
    main()
