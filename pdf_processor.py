import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import json
import hashlib
import numpy as np
from typing import List

class PDFProcessor:
    def __init__(self, pdf_dir="/Users/brian/pruning_paper_list"):
        self.pdf_dir = pdf_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        # 使用 sentence-transformers 作為嵌入模型
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        print("\nPDFProcessor 初始化:")
        print("PDF 目錄:", self.pdf_dir)
        
        # 初始化 ChromaDB 客戶端
        self.client = chromadb.Client()
        
        # 用於存儲每個文件對應的集合
        self.collections = {}
        
        # 用於存儲已處理文件的緩存
        self.cache_file = "processed_files_cache.json"
        self.processed_files = self._load_cache()
        self.mmr_lambda = 0.7  # MMR 參數，控制相關性和多樣性的平衡

    def _calculate_file_hash(self, file_path):
        """計算文件的 MD5 hash 值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _load_cache(self):
        """加載已處理文件的緩存"""
        try:
            if os.path.exists(self.cache_file) and os.path.getsize(self.cache_file) > 0:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"警告: 緩存文件讀取失敗 ({str(e)})，將創建新的緩存文件")
            # 確保緩存目錄存在
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # 創建新的緩存文件
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
        return {}

    def _save_cache(self):
        """保存已處理文件的緩存"""
        try:
            # 確保緩存目錄存在
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # 保存緩存
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"警告: 緩存文件保存失敗 ({str(e)})")

    def _load_chunks_from_cache(self, file_hash):
        """從緩存中加載已處理的 chunks"""
        if os.path.exists('rag_chunk_file1.json'):
            with open('rag_chunk_file1.json', 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)
                for filename, data in all_chunks.items():
                    if self.processed_files.get(filename) == file_hash:
                        return data
        return None

    def _extract_pdf_metadata(self, doc):
        """提取PDF的元數據（標題、作者等）"""
        metadata = doc.metadata
        title = metadata.get('title', '')
        authors = metadata.get('author', '')
        
        # 從第一頁提取文本
        first_page = doc[0]
        text = first_page.get_text()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 如果元數據中沒有標題和作者，嘗試從文本中提取
        if not title or not authors:
            for i, line in enumerate(lines[:5]):  # 只檢查前5行
                # 跳過空行
                if not line:
                    continue
                    
                # 第一個非空行通常是標題
                if not title and i == 0:
                    title = line
                    continue
                
                # 檢查是否是作者行（通常包含多個人名，用逗號分隔）
                if not authors and ',' in line and not any(keyword in line.lower() for keyword in 
                    ['abstract', 'introduction', 'university', 'department']):
                    # 確保行中包含多個大寫字母（表示可能是人名）
                    if sum(1 for c in line if c.isupper()) > 2:
                        authors = line
                        break
                
                # 如果找到 "ABSTRACT" 關鍵字，停止搜索
                if 'ABSTRACT' in line.upper():
                    break
        
        # 清理作者字符串（移除機構名稱等）
        if authors:
            # 分割作者字符串，通常作者之間用逗號分隔
            author_list = [author.strip() for author in authors.split(',')]
            # 過濾掉不像人名的部分（例如機構名稱、郵箱等）
            author_list = [author for author in author_list if 
                         not any(keyword in author.lower() for keyword in 
                               ['university', 'department', '@', 'institute', 'lab'])]
            authors = ', '.join(author_list)
        
        return {
            'title': title.strip(),
            'authors': authors.strip()
        }

    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text_chunks = []
        metadatas = []
        ids = []
        chunk_id = 0
        
        # 提取PDF元數據
        pdf_metadata = self._extract_pdf_metadata(doc)
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            chunks = self.text_splitter.split_text(text)
            
            for chunk in chunks:
                text_chunks.append(chunk)
                metadatas.append({
                    "page": page_num + 1,
                    "source": os.path.basename(pdf_path),
                    "title": pdf_metadata['title'],
                    "authors": pdf_metadata['authors']
                })
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
        
        doc.close()
        return text_chunks, metadatas, ids

    def _get_collection_name(self, filename):
        """根據文件名生成集合名稱"""
        # 移除 .pdf 後綴，替換特殊字符
        return filename.replace('.pdf', '').replace(' ', '_').lower()

    def process_all_pdfs(self):
        """處理所有 PDF 文件，每個文件創建獨立的集合"""
        all_chunks_data = {}
        any_new_files = False
        
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith('.pdf'):
                collection_name = self._get_collection_name(filename)
                print(f"\n處理文件: {filename}")
                print(f"集合名稱: {collection_name}")
                
                # 獲取或創建該文件的集合
                try:
                    collection = self.client.get_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    print(f"找到現有集合: {collection_name}")
                except:
                    collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    print(f"創建新集合: {collection_name}")
                
                self.collections[filename] = collection
                
                # 處理文件內容
                pdf_path = os.path.join(self.pdf_dir, filename)
                current_hash = self._calculate_file_hash(pdf_path)
                
                # 檢查集合中是否已有數據
                try:
                    collection_count = collection.count()
                    print(f"集合 {collection_name} 中的文檔數量: {collection_count}")
                except Exception as e:
                    print(f"獲取集合數量時出錯: {str(e)}")
                    collection_count = 0
                
                # 如果集合為空，需要添加數據
                if collection_count == 0:
                    print(f"集合 {collection_name} 為空，需要添加數據")
                    
                    # 檢查是否有緩存數據
                    if filename in self.processed_files and self.processed_files[filename] == current_hash:
                        cached_data = self._load_chunks_from_cache(current_hash)
                        if cached_data:
                            print(f"使用緩存數據添加到集合: {filename}")
                            try:
                                collection.add(
                                    documents=cached_data["chunks"],
                                    metadatas=cached_data["metadatas"],
                                    ids=cached_data["ids"]
                                )
                                print(f"成功從緩存添加 {len(cached_data['chunks'])} 個文檔到集合")
                                all_chunks_data[filename] = cached_data
                                continue
                            except Exception as e:
                                print(f"從緩存添加數據到集合時出錯: {str(e)}")
                    
                    # 如果沒有緩存或添加失敗，重新處理文件
                    print(f"處理文件: {filename}")
                    chunks, metadatas, ids = self.process_pdf(pdf_path)
                    
                    try:
                        collection.add(
                            documents=chunks,
                            metadatas=metadatas,
                            ids=ids
                        )
                        print(f"成功添加 {len(chunks)} 個文檔到集合")
                    except Exception as e:
                        print(f"添加數據到集合時出錯: {str(e)}")
                    
                    all_chunks_data[filename] = {
                        "chunks": chunks,
                        "metadatas": metadatas,
                        "ids": ids
                    }
                    
                    self.processed_files[filename] = current_hash
                    any_new_files = True
                else:
                    # 集合已有數據，使用緩存
                    if filename in self.processed_files and self.processed_files[filename] == current_hash:
                        cached_data = self._load_chunks_from_cache(current_hash)
                        if cached_data:
                            print(f"使用緩存數據: {filename}")
                            all_chunks_data[filename] = cached_data
                            continue
        
        # 保存緩存
        if any_new_files:
            with open('rag_chunk_file1.json', 'w', encoding='utf-8') as f:
                json.dump(all_chunks_data, f, ensure_ascii=False, indent=4)
            self._save_cache()
            print("已將所有 chunks 保存到 rag_chunk_file1.json")
        else:
            print("所有文件都是最新的，無需更新")

    def query(self, question, n_results=3, specific_file=None):
        """從指定文件的集合中查詢"""
        print("\nPDF Processor - 查詢參數:")
        print("問題:", question)
        print("指定文件:", specific_file)
        
        if specific_file:
            # 從指定文件的集合中查詢
            collection_name = self._get_collection_name(specific_file)
            try:
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                print(f"\n從集合 {collection_name} 中查詢")
                
                # 檢查集合中的文檔數量
                collection_count = collection.count()
                print(f"集合 {collection_name} 中的文檔數量: {collection_count}")
                
                if collection_count == 0:
                    print(f"警告: 集合 {collection_name} 為空，無法查詢")
                    return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
                
                results = collection.query(
                    query_texts=[question],
                    n_results=min(n_results, collection_count)
                )
                print(f"檢索到的文檔數量: {len(results['documents'][0])}")
                return results
                
            except Exception as e:
                print(f"查詢集合 {collection_name} 時出錯: {str(e)}")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        else:
            # 如果沒有指定文件，合併所有集合的結果
            print("\n搜索所有文件")
            all_results = []
            
            if not self.collections:
                print("警告: 沒有可用的集合")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            for filename, collection in self.collections.items():
                try:
                    # 檢查集合中的文檔數量
                    collection_count = collection.count()
                    print(f"集合 {collection.name} 中的文檔數量: {collection_count}")
                    
                    if collection_count == 0:
                        print(f"跳過空集合: {collection.name}")
                        continue
                    
                    results = collection.query(
                        query_texts=[question],
                        n_results=min(n_results, collection_count)
                    )
                    
                    if results["documents"][0]:
                        print(f"從集合 {collection.name} 檢索到 {len(results['documents'][0])} 個文檔")
                        all_results.extend(zip(
                            results["documents"][0],
                            results["metadatas"][0],
                            results["distances"][0]
                        ))
                except Exception as e:
                    print(f"查詢文件 {filename} 時出錯: {str(e)}")
            
            if not all_results:
                print("警告: 所有集合查詢結果為空")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            # 按相似度排序並取前 n_results 個結果
            all_results.sort(key=lambda x: x[2])  # 按距離排序
            top_results = all_results[:n_results]
            
            # 重新組織結果格式
            return {
                "documents": [[r[0] for r in top_results]],
                "metadatas": [[r[1] for r in top_results]],
                "distances": [[r[2] for r in top_results]]
            }

    def generate_answer(self, question, retrieved_context, model_provider="gemini"):
        """根據檢索到的上下文生成答案"""
        print("\n生成答案...")
        
        # 組織提示詞
        prompt = f"""
        你是一個專業的學術助手。請根據提供的上下文回答問題。
        
        上下文：
        {retrieved_context}
        
        問題：{question}
        
        回答要求：
        1. 只使用上下文中的信息回答問題
        2. 如果上下文中沒有足夠信息，請明確說明
        3. 提供具體的引用來源
        4. 保持學術性的回答風格
        """
        
        # 這裡我們只返回提示詞，實際的 LLM 調用在 qa_system.py 中進行
        return {
            "prompt": prompt,
            "question": question,
            "context": retrieved_context
        }

    def query_and_generate(self, question, n_results=5, specific_file=None):
        """檢索相關文檔並生成答案"""
        # 先進行檢索
        results = self.query(question, n_results, specific_file)
        
        # 如果沒有檢索到文檔，返回空結果
        if not results["documents"][0]:
            return {
                "answer_info": {
                    "prompt": f"請回答以下問題，但說明沒有找到相關文檔：{question}",
                    "question": question,
                    "context": ""
                },
                "documents": [],
                "metadatas": [],
                "distances": []
            }
        
        # 構建上下文
        context_with_citations = []
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            # 添加文件信息和引用
            source_info = f"[{i+1}] 文件: {meta.get('source', '')}, 頁碼: {meta.get('page', '')}"
            title_info = f"標題: {meta.get('title', '')}"
            author_info = f"作者: {meta.get('authors', '')}"
            
            context_with_citations.append(f"{source_info}\n{title_info}\n{author_info}\n\n{doc}\n")
        
        retrieved_context = "\n".join(context_with_citations)
        
        # 生成答案信息
        answer_info = self.generate_answer(question, retrieved_context)
        
        # 返回完整結果
        return {
            "answer_info": answer_info,
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0] if "distances" in results else []
        }



