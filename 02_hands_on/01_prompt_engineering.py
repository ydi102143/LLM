"""
プロンプトエンジニアリングの実践例

このスクリプトでは、効果的なプロンプトの書き方を学びます。
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

class PromptEngineer:
    def __init__(self, model_name="gpt2"):
        """プロンプトエンジニアリング用のクラス"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt, max_length=100, temperature=0.7, **kwargs):
        """テキスト生成"""
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
    
    def few_shot_example(self, task_description, examples, query):
        """Few-shot学習の例"""
        prompt = f"{task_description}\n\n"
        
        for example in examples:
            prompt += f"例: {example['input']}\n"
            prompt += f"回答: {example['output']}\n\n"
        
        prompt += f"問題: {query}\n回答:"
        return prompt
    
    def chain_of_thought(self, problem):
        """Chain of Thought（思考の連鎖）プロンプト"""
        prompt = f"""問題を段階的に考えて解決してください。

問題: {problem}

解決手順:
1. まず、問題を理解する
2. 必要な情報を整理する
3. 段階的に解決策を考える
4. 最終的な答えを導く

解答:"""
        return prompt
    
    def role_playing(self, role, task, context=""):
        """役割を指定したプロンプト"""
        prompt = f"""あなたは{role}です。{task}
{context}

回答:"""
        return prompt

def main():
    # プロンプトエンジニアの初期化
    pe = PromptEngineer()
    
    print("=== プロンプトエンジニアリングの実践例 ===\n")
    
    # 1. 基本的なプロンプト
    print("1. 基本的なプロンプト")
    basic_prompt = "The benefits of renewable energy are"
    result = pe.generate(basic_prompt, max_length=80)
    print(f"プロンプト: {basic_prompt}")
    print(f"結果: {result}\n")
    
    # 2. Few-shot学習
    print("2. Few-shot学習の例")
    task_desc = "感情分析を行ってください。入力された文章の感情をpositive、negative、neutralのいずれかで分類してください。"
    examples = [
        {"input": "I love this movie!", "output": "positive"},
        {"input": "This is terrible.", "output": "negative"},
        {"input": "The weather is okay.", "output": "neutral"}
    ]
    query = "I'm so excited about the new project!"
    
    few_shot_prompt = pe.few_shot_example(task_desc, examples, query)
    result = pe.generate(few_shot_prompt, max_length=50)
    print(f"Few-shotプロンプト:\n{few_shot_prompt}")
    print(f"結果: {result}\n")
    
    # 3. Chain of Thought
    print("3. Chain of Thought（思考の連鎖）")
    problem = "A store has 120 apples. They sell 30% of them in the morning and 40% of the remaining in the afternoon. How many apples are left?"
    cot_prompt = pe.chain_of_thought(problem)
    result = pe.generate(cot_prompt, max_length=150)
    print(f"Chain of Thoughtプロンプト:\n{cot_prompt}")
    print(f"結果: {result}\n")
    
    # 4. 役割指定
    print("4. 役割指定プロンプト")
    role_prompt = pe.role_playing(
        role="経験豊富なデータサイエンティスト",
        task="機械学習プロジェクトの成功要因について説明してください。",
        context="初心者にも分かりやすく、実践的なアドバイスを含めてください。"
    )
    result = pe.generate(role_prompt, max_length=120)
    print(f"役割指定プロンプト:\n{role_prompt}")
    print(f"結果: {result}\n")
    
    # 5. プロンプトの比較実験
    print("5. プロンプトの比較実験")
    base_query = "How to learn machine learning?"
    
    prompts = {
        "シンプル": base_query,
        "具体的": f"Give me a step-by-step guide on how to learn machine learning as a beginner. Include resources and timeline.",
        "例示付き": f"Explain how to learn machine learning. For example, start with Python basics, then learn libraries like scikit-learn, and practice with real datasets.",
        "構造化": f"""How to learn machine learning:

1. Prerequisites:
2. Learning path:
3. Practical projects:
4. Resources:
5. Timeline:

Please provide detailed guidance for each section."""
    }
    
    for style, prompt in prompts.items():
        print(f"\n--- {style}スタイル ---")
        result = pe.generate(prompt, max_length=100, temperature=0.7)
        print(f"プロンプト: {prompt}")
        print(f"結果: {result}")

if __name__ == "__main__":
    main()
