import os
from openai import OpenAI
import google.generativeai as genai
from pdf_processor import PDFProcessor
from typing import List, Dict

class QASystem:
    def __init__(self, model_provider="openai"):
        """
        初始化 QA 系統
        :param model_provider: 選擇使用的模型提供者 ("openai" 或 "gemini")
        """
        self.pdf_processor = PDFProcessor()
        self.model_provider = model_provider.lower()
        self.conversation_history = []  # 存儲對話歷史
        
        if self.model_provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.model_provider == "gemini":
            genai.configure(api_key="Your API Key")
            # 設置默認生成參數
            generation_config = {
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
            }
            
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-latest",
                generation_config=generation_config
            )
        else:
            raise ValueError("不支持的模型提供者。請選擇 'openai' 或 'gemini'")
        
    def initialize(self):
        """處理所有PDF文件並建立索引"""
        self.pdf_processor.process_all_pdfs()
        
    def _rerank_chunks(self, chunks: List[str], distances: List[float], top_k: int = 3) -> List[int]:
        """根據相似度得分重新排序chunks"""
        chunk_scores = list(enumerate(distances))
        chunk_scores.sort(key=lambda x: x[1])  # 按距離排序
        return [idx for idx, _ in chunk_scores[:top_k]]
    
    def _format_academic_citation(self, metadata: Dict) -> str:
        """將metadata格式化為學術引用格式"""
        return f"(來源: {metadata['source']}, 頁碼: {metadata['page']})"
    
    def _get_answer_from_model(self, messages, context, question):
        """根據不同的模型提供者獲取答案"""
        if self.model_provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        else:  # gemini
            # 將 messages 轉換為 Gemini 格式
            prompt = f"""
            系統: 你是一個專業的學術論文助手。請根據提供的論文內容回答問題，並：
            1. 提供具體的引用來源
            2. 如果無法從內容中找到答案，請明確說明
            3. 如果有相關的後續問題建議，請在回答後列出
            4. 保持學術性的回答風格

            內容: {context}

            問題: {question}
            """
            
            # 如果有對話歷史，添加到提示中
            if self.conversation_history:
                history = "\n\n歷史對話:\n"
                for msg in self.conversation_history[-3:]:  # 只使用最近3輪
                    role = "用戶" if msg["role"] == "user" else "助手"
                    history += f"{role}: {msg['content']}\n"
                prompt += f"\n{history}"

            response = self.model.generate_content(prompt)
            return response.text

    def _get_specific_file(self, question: str) -> str:
        """根據問題內容選擇特定文件"""
        question = question.lower()
        
        # 文件匹配規則
        file_patterns = {
            "PRUNING FILTERS FOR EFFICIENT CONVNETS.pdf": [
                "pruning filters", "efficient convnets", "li", "kadav"
            ],
            "NEURAL PRUNING VIA GROWING REGULARIZATION.pdf": [
                "growing regularization", "greg", "neural pruning"
            ],
            "Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.pdf": [
                "depgraph", "structural pruning", "fang"
            ],
            "PRUNING CONVOLUTIONAL NEURAL NETWORKS.pdf": [
                "pruning convolutional", "molchanov"
            ],
            "Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf": [
                "learning efficient", "liu"
            ],
            "NIPS-1989-optimal-brain-damage-Paper.pdf": [
                "optimal brain", "brain damage", "obd", "optimal brain damage","HessianImportance"
            ]
        }
        
        # 檢查每個文件的關鍵字
        for file_name, keywords in file_patterns.items():
            if any(keyword in question for keyword in keywords):
                return file_name
        
        return None

    def get_answer(self, question: str) -> dict:
        # 獲取所有可用的 PDF 文件列表
        available_files = [f for f in os.listdir(self.pdf_processor.pdf_dir) if f.endswith('.pdf')]
        print("\n可用文件列表:", available_files)
        
        # 根據問題內容選擇特定文件
        specific_file = self._get_specific_file(question)
        print("\n選擇的文件:", specific_file)
        
        # 使用 RAG 進行檢索並生成答案信息
        rag_results = self.pdf_processor.query_and_generate(
            question=question,
            n_results=5,
            specific_file=specific_file
        )
        
        # 使用 LLM 生成最終答案
        answer = self._get_answer_from_model(
            messages=[{
                "role": "system", 
                "content": "你是一個專業的學術論文助手。請根據提供的論文內容回答問題。"
            }, {
                "role": "user", 
                "content": rag_results["answer_info"]["prompt"]
            }],
            context=rag_results["answer_info"]["context"],
            question=question
        )
        
        # 更新對話歷史
        self.conversation_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        
        # 返回結果
        return {
            "Question": question,
            "Answer": answer,
            "rag_chunks": rag_results["documents"],
            "references": [{
                "title": meta["title"],
                "authors": meta["authors"],
                "source": meta["source"],
                "page": meta["page"]
            } for meta in rag_results["metadatas"]],
            "used_rag": True,
            "selected_file": specific_file,
            "conversation_history": self.conversation_history
        }
    
    def clear_history(self):
        """清除對話歷史"""
        self.conversation_history = [] 

