import os
from langchain_ollama import OllamaEmbeddings, ChatOllama # 理解與尋找、回答與總結
from langchain_chroma import Chroma # 向量資料庫
from langchain_core.prompts import ChatPromptTemplate # 提示詞模板
from langchain_core.output_parsers import StrOutputParser # 輸出解析器

# --- 1. 設定區  ---
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"  
PERSIST_DIRECTORY = "./db_ollama"
k_value=5

# --- 2. 檢查資料庫是否存在 ---
if not os.path.exists(PERSIST_DIRECTORY):
    print(f"❌ 找不到資料庫目錄: {PERSIST_DIRECTORY}，請先執行建庫程式。")
    exit()

# --- 3. 初始化地端模型 ---
print(f"正在載入地端模型 ({LLM_MODEL})...")

# Embedding 模型 (用來把問題轉成向量去搜尋)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# LLM 模型 (用來閱讀搜尋結果並回答)

llm = ChatOllama(
    model=LLM_MODEL, 
    temperature=0,
    num_gpu=1  # 設為 1 通常代表啟用 GPU 加速
)

# 載入現有資料庫
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# --- 4. 設定 Prompt (針對地端模型優化) ---
prompt = ChatPromptTemplate.from_template("""
你是一個專業的醫療報告分析助手。請根據下方提供的報告內容來回答問題。
答案請使用「繁體中文」回答。如果你無法從內容中找到答案，請回答不知道，不要胡編亂造。

<報告內容>
{context}
</報告內容>

問題：{input}
""")

# 建立執行鏈
chain = prompt | llm | StrOutputParser()

# --- 5. 執行提問 ---
query = "這份診斷報告的報告撰寫人是誰？" 

print(f"\n🔍 正在檢索資料並生成回答...")
# 檢索最相關的 3 個片段
docs = db.similarity_search(query, k=k_value)

# 執行
try:
    result = chain.invoke({
        "input": query, 
        "context": docs
    })
    print("\n" + "="*30)
    print(f"問：{query}")
    print(f"答：\n{result}")
    print("="*30)
except Exception as e:
    print(f"❌ 執行失敗: {e}")
    print("提示：請確保 Ollama 伺服器正在執行中。")
