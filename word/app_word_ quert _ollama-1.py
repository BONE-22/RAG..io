import os
import sys
import time
import torch
import warnings 
from tqdm import tqdm
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore", category=DeprecationWarning) 

# --- 1. 設定區 ---
FORCE_DEVICE = "cuda"  # "cpu" 或 "cuda"
FILE_PATH = "貿特198診斷報告Final.docx"
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "./db_ollama"

if not os.path.exists(FILE_PATH):
    print(f"❌ 找不到檔案: {FILE_PATH}")
    sys.exit()

# --- 2. 硬體判定 ---
if FORCE_DEVICE == "cuda" and torch.cuda.is_available():
    current_device = "GPU (CUDA)"
elif FORCE_DEVICE == "cpu":
    current_device = "CPU"
else:
    current_device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
print(f"目前適用裝置: {current_device}")

# --- 3. 載入與切分文件 ---
loader = Docx2txtLoader(FILE_PATH)
raw_docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
all_splits = text_splitter.split_documents(raw_docs)
total_chunks = len(all_splits)

# --- 4. 初始化地端 Embedding 模型 ---
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

print(f"\n正在進行地端向量化 (總計 {total_chunks} 個段落)...")
start_time = time.time()

try:
    # 建立空的資料庫
    vectorstore = Chroma.from_documents(
        documents=[all_splits[0]], 
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # 分批處理並顯示進度條
    batch_size = 5
    for i in tqdm(range(1, total_chunks, batch_size), desc=f"地端 {current_device} 處理中"):
        batch = all_splits[i : i + batch_size]
        vectorstore.add_documents(batch)
    
    duration = time.time() - start_time
    
    print(f"✅ 地端向量資料庫建置完成！")
    print(f"⏱️ 總耗時: {duration:.2f} 秒")

except Exception as e:
    print(f"❌ 發生錯誤: {e}")
    print("提示：請檢查 Ollama 是否已啟動，且已執行過 'ollama pull nomic-embed-text'")
