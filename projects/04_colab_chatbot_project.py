"""
チャットボットプロジェクト（Google Colab版）

このプロジェクトでは、Google Colab環境で実用的なチャットボットアプリケーションを開発します。
Colab環境に最適化された実装になっています。
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ColabChatbotProject:
    """Colab用チャットボットプロジェクトクラス"""
    
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.conversation_history = []
        self.system_prompt = "You are a helpful and friendly AI assistant."
        self.max_history = 5  # Colab用に削減
        self.max_length = 100  # Colab用に削減
        self.temperature = 0.7
        self.colab_optimized = True
    
    def setup_model(self):
        """モデルのセットアップ（Colab最適化）"""
        print("🔄 チャットボットモデルをセットアップ中...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ モデル '{self.model_name}' の読み込み完了")
            print(f"   パラメータ数: {self.model.num_parameters():,}")
            print(f"   語彙数: {len(self.tokenizer)}")
            
            # Colab環境でのメモリ使用量表示
            if torch.cuda.is_available():
                print(f"   GPU メモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            return True
            
        except Exception as e:
            print(f"❌ モデルの読み込みに失敗: {e}")
            print("軽量なモデルを試してください")
            return False
    
    def add_to_history(self, role: str, content: str):
        """会話履歴に追加（Colab最適化）"""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # 履歴が長すぎる場合は古いものを削除（Colab用に制限）
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def format_conversation(self, user_input: str) -> str:
        """会話をフォーマット（Colab最適化）"""
        # システムプロンプト
        formatted = f"{self.system_prompt}\\n\\n"
        
        # 会話履歴（Colab用に制限）
        for message in self.conversation_history[-3:]:  # 最新3件のみ
            if message['role'] == 'user':
                formatted += f"User: {message['content']}\\n"
            elif message['role'] == 'assistant':
                formatted += f"Assistant: {message['content']}\\n"
        
        # 現在のユーザー入力
        formatted += f"User: {user_input}\\nAssistant:"
        
        return formatted
    
    def generate_response(self, user_input: str) -> str:
        """応答を生成（Colab最適化）"""
        if self.model is None or self.tokenizer is None:
            return "モデルが読み込まれていません。"
        
        try:
            # 会話をフォーマット
            formatted_conversation = self.format_conversation(user_input)
            
            # トークン化
            inputs = self.tokenizer.encode(formatted_conversation, return_tensors="pt")
            
            # 生成（Colab用に制限）
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # デコード
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 応答部分のみを抽出
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            else:
                response = response[len(formatted_conversation):].strip()
            
            # 応答をクリーニング
            response = self.clean_response(response)
            
            return response
            
        except Exception as e:
            return f"エラーが発生しました: {str(e)}"
    
    def clean_response(self, response: str) -> str:
        """応答をクリーニング（Colab最適化）"""
        # 不要な部分を削除
        if "User:" in response:
            response = response.split("User:")[0].strip()
        
        # 空の応答の場合はデフォルトメッセージ
        if not response or len(response.strip()) < 3:
            response = "申し訳ございませんが、適切な応答を生成できませんでした。もう一度お試しください。"
        
        return response.strip()
    
    def chat(self, user_input: str) -> str:
        """チャット機能（Colab最適化）"""
        # ユーザー入力を履歴に追加
        self.add_to_history('user', user_input)
        
        # 応答を生成
        response = self.generate_response(user_input)
        
        # 応答を履歴に追加
        self.add_to_history('assistant', response)
        
        return response
    
    def start_interactive_chat(self):
        """インタラクティブチャットを開始（Colab最適化）"""
        print("\n🤖 チャットボットが起動しました！（Colab版）")
        print("=" * 50)
        print("チャットを開始してください。")
        print("終了するには 'quit' と入力してください。")
        print("履歴をクリアするには 'clear' と入力してください。")
        print("設定を変更するには 'settings' と入力してください。")
        
        # デモ用の会話
        demo_inputs = [
            "こんにちは！",
            "あなたは何ができますか？",
            "今日の天気について教えてください",
            "ありがとうございました",
            "さようなら"
        ]
        
        print("\n🎭 デモ会話を開始します")
        for user_input in demo_inputs:
            print(f"\n👤 あなた: {user_input}")
            response = self.chat(user_input)
            print(f"🤖 ボット: {response}")
        
        print("\n✅ デモ会話が完了しました")
        print("実際の会話を試すには、上記のコードを実行してください")
    
    def show_settings(self):
        """設定を表示・変更（Colab最適化）"""
        print("\n⚙️ チャットボット設定（Colab版）")
        print("=" * 30)
        print(f"現在の設定:")
        print(f"  モデル: {self.model_name}")
        print(f"  最大履歴数: {self.max_history}")
        print(f"  最大生成長: {self.max_length}")
        print(f"  温度: {self.temperature}")
        print(f"  システムプロンプト: {self.system_prompt}")
        print(f"  Colab最適化: {self.colab_optimized}")
        
        print("\n変更可能な設定:")
        print("1. 最大履歴数")
        print("2. 最大生成長")
        print("3. 温度")
        print("4. システムプロンプト")
        print("5. 戻る")
        
        print("\n💡 Colab環境での制限:")
        print("• メモリ制限により、一部の設定は制限されています")
        print("• 定期的にランタイムを再起動してメモリをクリア")
        print("• 大きなモデルは段階的に読み込み")
    
    def export_conversation(self, filename="colab_conversation_history.json"):
        """会話履歴をエクスポート（Colab最適化）"""
        print(f"\n💾 会話履歴をエクスポート中: {filename}")
        
        export_data = {
            'model_name': self.model_name,
            'settings': {
                'max_history': self.max_history,
                'max_length': self.max_length,
                'temperature': self.temperature,
                'system_prompt': self.system_prompt,
                'colab_optimized': self.colab_optimized
            },
            'conversation_history': self.conversation_history,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 会話履歴を {filename} に保存しました")
        print("💡 Google Driveに保存するには、ファイルをドラッグ&ドロップしてください")
    
    def load_conversation(self, filename="colab_conversation_history.json"):
        """会話履歴を読み込み（Colab最適化）"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.conversation_history = data.get('conversation_history', [])
            
            # 設定を復元
            settings = data.get('settings', {})
            self.max_history = settings.get('max_history', self.max_history)
            self.max_length = settings.get('max_length', self.max_length)
            self.temperature = settings.get('temperature', self.temperature)
            self.system_prompt = settings.get('system_prompt', self.system_prompt)
            self.colab_optimized = settings.get('colab_optimized', self.colab_optimized)
            
            print(f"✅ 会話履歴を {filename} から読み込みました")
            print(f"   履歴数: {len(self.conversation_history)}件")
            print(f"   Colab最適化: {self.colab_optimized}")
            
        except FileNotFoundError:
            print(f"❌ ファイル {filename} が見つかりません")
        except Exception as e:
            print(f"❌ 読み込み中にエラーが発生しました: {e}")
    
    def demo_conversation(self):
        """デモ会話を実行（Colab最適化）"""
        print("\n🎭 デモ会話を開始します（Colab版）")
        print("=" * 40)
        
        demo_inputs = [
            "こんにちは！",
            "あなたは何ができますか？",
            "今日の天気について教えてください",
            "ありがとうございました",
            "さようなら"
        ]
        
        for user_input in demo_inputs:
            print(f"\n👤 あなた: {user_input}")
            response = self.chat(user_input)
            print(f"🤖 ボット: {response}")
        
        print("\n✅ デモ会話が完了しました")
    
    def run_complete_project(self):
        """完全なプロジェクトの実行（Colab最適化）"""
        print("🚀 チャットボットプロジェクトを開始します（Colab版）")
        print("=" * 50)
        
        # 1. モデルのセットアップ
        if not self.setup_model():
            print("❌ プロジェクトを終了します")
            return
        
        # 2. デモ会話
        self.demo_conversation()
        
        # 3. インタラクティブチャット
        self.start_interactive_chat()
        
        # 4. 会話履歴のエクスポート
        self.export_conversation()
        
        print("\n✅ プロジェクトが完了しました！")
        print("チャットボットアプリケーションが正常に動作しています。")
        print("\n💡 Colab環境でのアドバイス:")
        print("• 定期的にランタイムを再起動してメモリをクリア")
        print("• 結果はGoogle Driveに保存")
        print("• エラーが発生したら設定を軽量化")
        print("• 大きなモデルは段階的に読み込み")

def main():
    """メイン関数"""
    print("チャットボットプロジェクト（Google Colab版）")
    print("=" * 40)
    
    # プロジェクトの初期化
    chatbot = ColabChatbotProject()
    
    # 完全なプロジェクトの実行
    chatbot.run_complete_project()

if __name__ == "__main__":
    main()
