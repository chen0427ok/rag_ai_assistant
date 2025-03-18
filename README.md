# PDF 問答系統

本專案是一個基於 Python 和 AI 模型（OpenAI GPT-4 或 Google Gemini）的 PDF 問答系統，能夠自動處理 PDF 文件，並透過檢索增強生成（RAG）技術回答用戶問題。

## 功能特點

- **PDF 處理**
  - 自動解析 PDF 文件內容
  - 提取標題、作者等元數據
  - 將文本分割為 chunk，便於索引與查詢

- **語義檢索**
  - 使用 ChromaDB 進行向量化儲存和檢索
  - 支持特定文件和全局檢索
  - 提供相關文獻來源與引用資訊

- **AI 問答系統**
  - 支持 OpenAI GPT-4 和 Google Gemini 模型
  - 提供具學術引用格式的答案
  - 支援多輪對話，並保持上下文連貫

- **管理與緩存**
  - 自動處理並建立 PDF 索引
  - 緩存處理結果，提高重複查詢效率

---

## 技術棧

- **程式語言**: Python 3.11
- **向量資料庫**: ChromaDB
- **嵌入模型**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **AI 提供者**: OpenAI GPT-4 / Google Gemini
- **其他工具**:
  - PyMuPDF (fitz)
  - LangChain

---

## 安裝指南

### 1. 克隆專案

```bash
git clone https://github.com/your-username/pdf-qa-system.git
cd pdf-qa-system
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 設置環境變數

創建 `.env` 文件，並填入以下資訊：

```env
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
PDF_DIRECTORY=/path/to/pdf_files
```

### 4. 執行程式

```bash
python main.py
```

---

## 專案結構

```
pdf-qa-system/
├── main.py                  # 主程式
├── qa_system.py             # 問答系統模組
├── pdf_processor.py         # PDF 處理模組
├── processed_files_cache.json # 處理緩存
├── QA.json                  # 問答結果儲存
├── rag_chunk_file1.json     # PDF chunk 緩存
└── requirements.txt         # Python 依賴
```

---

## 使用方法

1. **啟動系統**：執行 `main.py`。
2. **提問**：在提示輸入問題。
3. **查閱答案**：系統將提供答案及參考文獻。
4. **管理對話**：可清除歷史紀錄或結束對話。

---

## 開發者

- [陳致希](https://github.com/chen0427)

## 授權

本專案採用 MIT License 開源授權。

## 注意事項

- 勿將敏感資訊（如 API key）提交至版本控制。
- 定期更新依賴項，以確保安全性。

