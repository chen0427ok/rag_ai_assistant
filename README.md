# PDF 問答系統

本專案是一個基於 Flask、MongoDB 和 AI 模型（OpenAI GPT-4 或 Google Gemini）的 PDF 問答系統，能夠處理 PDF 文件，並通過檢索增強生成（RAG）技術回答用戶問題。

## 功能特點

- **PDF 處理**
  - 自動解析 PDF 文件，提取內容並索引
  - 提取標題、作者等元數據
  - 使用 Chunk 分割技術處理長文本
  
- **語義查詢**
  - 基於 ChromaDB 進行向量化存儲與檢索
  - 使用 MMR（最大邊際相關性）技術提高檢索質量
  - 可基於特定文件或全局檢索
  
- **AI 問答系統**
  - 支持 OpenAI GPT-4 或 Google Gemini 進行回答生成
  - 提供學術引用格式的答案
  - 支持多輪對話與歷史追蹤
  
- **系統管理**
  - 自動處理 PDF 文件並建立索引
  - 提供查詢 API 並支持多次對話
  - 清除歷史對話功能

---

## 技術棧

- **後端**: Python
- **框架**: Flask
- **數據庫**: MongoDB
- **嵌入模型**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **向量數據庫**: ChromaDB
- **AI 提供者**: OpenAI GPT-4 / Google Gemini

---

## 安裝指南

### 1. 克隆專案
```bash
git clone https://github.com/your-username/pdf-qa-system.git
cd pdf-qa-system
```

### 2. 建立虛擬環境並安裝依賴
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. 設定環境變數
創建 `.env` 文件，並填入以下內容：
```ini
MONGODB_URI=your_mongodb_uri
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
FLASK_SECRET_KEY=your_secret_key
PDF_DIRECTORY=/path/to/your/pdf_folder
```

### 4. 啟動系統
```bash
python main.py
```

---

## 專案結構
```
pdf-qa-system/
├── main.py              # 啟動主程式
├── qa_system.py         # 問答系統模組
├── pdf_processor.py     # PDF 處理模組
├── requirements.txt     # 依賴文件
├── .env.example         # 環境變數示例
├── QA.json              # 問答記錄文件
├── rag_chunk_file1.json # PDF 內容索引
```

---

## 使用方式

1. **執行系統**
   啟動 `main.py` 後，系統將自動處理 PDF 文件，並準備好回答用戶問題。

2. **輸入問題**
   在終端輸入問題，例如：
   ```
   請解釋 OBD（Optimal Brain Damage）方法的核心思想？
   ```

3. **獲取答案**
   系統將檢索相關文獻，並返回答案與引用來源。

4. **查看參考文獻**
   系統會提供 PDF 來源、頁碼、標題與作者資訊，確保答案的可驗證性。

5. **多輪對話**
   用戶可多次提問，系統會記錄對話歷史，提供上下文連貫的答案。

6. **清除對話歷史**
   若需要清除歷史記錄，可輸入：
   ```
   clear
   ```

7. **退出系統**
   若要結束對話，輸入：
   ```
   exit
   ```

---

## API 介面（未來計劃）

未來版本將提供 RESTful API，允許通過 HTTP 請求進行查詢。

---

## 參考文獻

- [ChromaDB](https://github.com/chroma-core/chroma)
- [Sentence-Transformers](https://www.sbert.net/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Google Gemini](https://ai.google.dev/)

---

## 開發者

- [Your Name](https://github.com/your-username)

## 授權

本專案基於 MIT License 開源，詳見 [LICENSE](LICENSE)。

## 注意事項

- 請勿將 `.env` 文件或 API Key 提交至 GitHub。
- 定期更新 `requirements.txt` 以修復安全漏洞。
- 建議使用 Python 3.11 以獲得最佳性能。

