import os
import torch #GPU 加速
from langchain_chroma import Chroma # 向量資料庫
from langchain_ollama import OllamaEmbeddings  # 向量化

# --- 1. 設定區 ---
FORCE_DEVICE = "cuda"  # 預期使用的裝置
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "./db_ollama"
k_value=5

# --- 2. 硬體判定與資訊獲取 ---
cuda_available = torch.cuda.is_available()

if FORCE_DEVICE == "cuda" and cuda_available:
    device_info = f"GPU (CUDA)"
elif FORCE_DEVICE == "cuda" and not cuda_available:
    device_info = "CPU (雖然設定為 cuda，但系統未偵測到可用 GPU，已切換至 CPU)"
else:
    device_info = "CPU"

# --- 3. 初始化 Embedding 模型 ---
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

if not os.path.exists(PERSIST_DIRECTORY):
    print(f"❌ 找不到資料庫目錄: {PERSIST_DIRECTORY}，請先執行建庫程式。")
    exit()

# --- 4. 載入本地存好的 Chroma 資料 ---
db = Chroma(
    persist_directory=PERSIST_DIRECTORY, 
    embedding_function=embeddings
)

# --- 5. 測試檢索 ---
query = "報告撰寫人是誰？"
# 檢索最相關的片段
docs = db.similarity_search(query, k=k_value)

# --- 6. 輸出結果 ---
print(f"【系統硬體資訊】: {device_info}")
print(f"【使用模型名稱】: {EMBED_MODEL}")
print(f"【資料庫路徑】: {PERSIST_DIRECTORY}")
print("-" * 30)
print(f"問題：{query}\n")
print("--- 檢索到的報告內容如下 ---")

for i, doc in enumerate(docs):
    clean_content = " ".join(doc.page_content.split())
    source = doc.metadata.get("source", "未知來源")
    print(f"[{i+1}] (來源: {source}) {clean_content}\n")
