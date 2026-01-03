from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

#讀取資料集
animal_data = pd.read_csv("animal-fun-facts-dataset.csv")

#初始化 Embedding Function ( 使用模型將文字轉為向量 )
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 準備 Metadata 
metadatas = []
for i, row in animal_data.iterrows():
    metadatas.append(
        {
            "Animal Name": row["animal_name"],
            "Source URL": row["source"],
            # "Media URL": row["media_link"],
            # "Wikipedia URL": row["wikipedia_link"],
        }
    )

# 確保文本欄位是字串格式 
animal_data["text"] = animal_data["text"].astype(str)

# 建立 FAISS 向量資料庫 
faiss = FAISS.from_texts(animal_data["text"].to_list(), embedding_function, metadatas)

query = "What is ship of the desert?"
k_count = 3  # 設定要找回前幾名結果
results = faiss.similarity_search_with_score(query, k=k_count)

print(f"問題: '{query}'")

# 使用 enumerate 來標示 K 值 (從 1 開始)
for i, (doc, score) in enumerate(results, 1):
    print(f"【排名 K = {i}】")
    print(f"相似度分數 : {score:.4f}")
    print(f"動物名稱: {doc.metadata.get('Animal Name')}")
    print(f"資料來源: {doc.metadata.get('Source URL')}")
