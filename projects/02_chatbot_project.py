"""
チャットボットプロジェクト

このプロジェクトでは、実用的なチャットボットアプリケーションを開発します。
会話履歴の管理、コンテキストの保持、適切な応答生成を実装します。
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

class ChatbotProject:
    """チャットボットプロジェクトクラス"""
    
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.conversation_history = []
        self.system_prompt = "You are a helpful and friendly AI assistant."
        self.max_history = 10  # 保持する会話履歴の最大数
        self.max_length = 150  # 生成する最大長
        self.temperature = 0.7  # 生成のランダム性
    
    def setup_model(self):
        """モデルのセットアップ"""
        print("🔄 チャットボットモデルをセットアップ中...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ モデル '{self.model_name}' の読み込み完了")
            print(f"   パラメータ数: {self.model.num_parameters():,}")
            print(f"   語彙数: {len(self.tokenizer)}")
            
            return True
            
        except Exception as e:
            print(f"❌ モデルの読み込みに失敗: {e}")
            return False
    
    def add_to_history(self, role: str, content: str):
        """会話履歴に追加"""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # 履歴が長すぎる場合は古いものを削除
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def format_conversation(self, user_input: str) -> str:
        """会話をフォーマット"""
        # システムプロンプト
        formatted = f"{self.system_prompt}\\n\\n"
        
        # 会話履歴
        for message in self.conversation_history:
            if message['role'] == 'user':
                formatted += f"User: {message['content']}\\n"
            elif message['role'] == 'assistant':
                formatted += f"Assistant: {message['content']}\\n"
        
        # 現在のユーザー入力
        formatted += f"User: {user_input}\\nAssistant:"
        
        return formatted
    
    def generate_response(self, user_input: str) -> str:
        """応答を生成"""
        if self.model is None or self.tokenizer is None:
            return "モデルが読み込まれていません。"
        
        try:
            # 会話をフォーマット
            formatted_conversation = self.format_conversation(user_input)
            
            # トークン化
            inputs = self.tokenizer.encode(formatted_conversation, return_tensors="pt")
            
            # 生成
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
        """応答をクリーニング"""
        # 不要な部分を削除
        if "User:" in response:
            response = response.split("User:")[0].strip()
        
        # 空の応答の場合はデフォルトメッセージ
        if not response or len(response.strip()) < 3:
            response = "申し訳ございませんが、適切な応答を生成できませんでした。もう一度お試しください。"
        
        return response.strip()
    
    def chat(self, user_input: str) -> str:
        """チャット機能"""
        # ユーザー入力を履歴に追加
        self.add_to_history('user', user_input)
        
        # 応答を生成
        response = self.generate_response(user_input)
        
        # 応答を履歴に追加
        self.add_to_history('assistant', response)
        
        return response
    
    def start_interactive_chat(self):
        """インタラクティブチャットを開始"""
        print("\\n🤖 チャットボットが起動しました！")
        print("=" * 50)
        print("チャットを開始してください。（終了するには 'quit' と入力）")
        print("履歴をクリアするには 'clear' と入力してください。")
        print("設定を変更するには 'settings' と入力してください。")
        
        while True:
            try:
                user_input = input("\\n👤 あなた: ")
                
                if user_input.lower() == 'quit':
                    print("\\n👋 チャットを終了します。お疲れ様でした！")
                    break
                
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("\\n🗑️ 会話履歴をクリアしました。")
                    continue
                
                elif user_input.lower() == 'settings':
                    self.show_settings()
                    continue
                
                elif not user_input.strip():
                    print("メッセージを入力してください。")
                    continue
                
                # 応答を生成
                response = self.chat(user_input)
                print(f"\\n🤖 ボット: {response}")
                
            except KeyboardInterrupt:
                print("\\n\\n👋 チャットを終了します。お疲れ様でした！")
                break
            except Exception as e:
                print(f"\\n❌ エラーが発生しました: {e}")
                continue
    
    def show_settings(self):
        """設定を表示・変更"""
        print("\\n⚙️ チャットボット設定")
        print("=" * 30)
        print(f"現在の設定:")
        print(f"  モデル: {self.model_name}")
        print(f"  最大履歴数: {self.max_history}")
        print(f"  最大生成長: {self.max_length}")
        print(f"  温度: {self.temperature}")
        print(f"  システムプロンプト: {self.system_prompt}")
        
        print("\\n変更可能な設定:")
        print("1. 最大履歴数")
        print("2. 最大生成長")
        print("3. 温度")
        print("4. システムプロンプト")
        print("5. 戻る")
        
        try:
            choice = input("\\n変更したい設定の番号を入力してください: ")
            
            if choice == '1':
                new_value = int(input(f"新しい最大履歴数 (現在: {self.max_history}): "))
                self.max_history = max(1, min(50, new_value))
                print(f"✅ 最大履歴数を {self.max_history} に変更しました。")
            
            elif choice == '2':
                new_value = int(input(f"新しい最大生成長 (現在: {self.max_length}): "))
                self.max_length = max(50, min(500, new_value))
                print(f"✅ 最大生成長を {self.max_length} に変更しました。")
            
            elif choice == '3':
                new_value = float(input(f"新しい温度 (現在: {self.temperature}): "))
                self.temperature = max(0.1, min(2.0, new_value))
                print(f"✅ 温度を {self.temperature} に変更しました。")
            
            elif choice == '4':
                new_prompt = input(f"新しいシステムプロンプト (現在: {self.system_prompt}): ")
                if new_prompt.strip():
                    self.system_prompt = new_prompt.strip()
                    print(f"✅ システムプロンプトを変更しました。")
            
            elif choice == '5':
                print("設定メニューを終了します。")
            
            else:
                print("無効な選択です。")
                
        except ValueError:
            print("無効な値です。")
        except Exception as e:
            print(f"エラーが発生しました: {e}")
    
    def export_conversation(self, filename="conversation_history.json"):
        """会話履歴をエクスポート"""
        print(f"\\n💾 会話履歴をエクスポート中: {filename}")
        
        export_data = {
            'model_name': self.model_name,
            'settings': {
                'max_history': self.max_history,
                'max_length': self.max_length,
                'temperature': self.temperature,
                'system_prompt': self.system_prompt
            },
            'conversation_history': self.conversation_history,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 会話履歴を {filename} に保存しました")
    
    def load_conversation(self, filename="conversation_history.json"):
        """会話履歴を読み込み"""
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
            
            print(f"✅ 会話履歴を {filename} から読み込みました")
            print(f"   履歴数: {len(self.conversation_history)}件")
            
        except FileNotFoundError:
            print(f"❌ ファイル {filename} が見つかりません")
        except Exception as e:
            print(f"❌ 読み込み中にエラーが発生しました: {e}")
    
    def demo_conversation(self):
        """デモ会話を実行"""
        print("\\n🎭 デモ会話を開始します")
        print("=" * 40)
        
        demo_inputs = [
            "こんにちは！",
            "あなたは何ができますか？",
            "今日の天気について教えてください",
            "ありがとうございました",
            "さようなら"
        ]
        
        for user_input in demo_inputs:
            print(f"\\n👤 あなた: {user_input}")
            response = self.chat(user_input)
            print(f"🤖 ボット: {response}")
        
        print("\\n✅ デモ会話が完了しました")
    
    def run_complete_project(self):
        """完全なプロジェクトの実行"""
        print("🚀 チャットボットプロジェクトを開始します")
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
        
        print("\\n✅ プロジェクトが完了しました！")
        print("チャットボットアプリケーションが正常に動作しています。")

def main():
    """メイン関数"""
    print("チャットボットプロジェクト")
    print("=" * 30)
    
    # プロジェクトの初期化
    chatbot = ChatbotProject()
    
    # 完全なプロジェクトの実行
    chatbot.run_complete_project()

if __name__ == "__main__":
    main()
