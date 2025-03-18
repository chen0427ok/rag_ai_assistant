import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 添加這行來解決 tokenizers 警告

from qa_system import QASystem
import json

def main():
    # 初始化 QA 系統
    qa_system = QASystem(model_provider="gemini")
    
    print("正在初始化系統，處理 PDF 文件中...")
    qa_system.initialize()
    
    # 多輪對話循環
    while True:
        # 獲取用戶輸入
        question = input("\n請輸入您的問題 (輸入 'exit' 結束對話，'clear' 清除歷史): ")
        
        if question.lower() == 'exit':
            break
            
        if question.lower() == 'clear':
            qa_system.clear_history()
            print("已清除對話歷史")
            continue
        
        print("\n正在生成答案...")
        result = qa_system.get_answer(question)
        
        # 輸出答案
        print("\n答案:")
        print(result["Answer"])
        
        # 顯示來源文獻信息
        print("\n參考文獻:")
        for i, ref in enumerate(result["references"], 1):
            print(f"{i}. 標題: {ref['title']}")
            print(f"   作者: {ref['authors']}")
            print(f"   來源: {ref['source']}, 頁碼: {ref['page']}\n")
        
        # 顯示是否使用了 RAG
        print(f"\n使用 RAG: {'是' if result['used_rag'] else '否'}")
        if result['used_rag'] and 'selected_file' in result:
            print(f"選擇的文件: {result['selected_file']}")
        
        # 保存當前問答結果
        qa_data = {
            "Question": result["Question"],
            "Answer": result["Answer"],
            "rag_chunks": result["rag_chunks"],
            "references": result["references"],
            "conversation_history": result["conversation_history"]
        }
        
        # 保存到 JSON 文件
        with open('QA.json', 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=4)
        print("\nQA 結果已保存到 QA.json")

if __name__ == "__main__":
    main() 

