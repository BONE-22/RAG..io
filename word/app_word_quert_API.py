from configparser import ConfigParser #讀取與管理設定檔
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI #Google提供Embedding模型
from langchain_chroma import Chroma #向量資料庫
from langchain_core.prompts import ChatPromptTemplate #提示詞模板
from langchain_core.output_parsers import StrOutputParser #輸出的訊息對象轉換成純字串

# --- 1. 設定區 ---
EMBED_MODEL = "models/text-embedding-004" 
PERSIST_DIRECTORY = "./chroma_db_medical"
k_value=5

# --- 2. 載入金鑰 ---
config = ConfigParser()
config.read("config.ini")
api_key = config["Gemini"]["API_KEY"]

# --- 3. 讀取現有資料庫 ---
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    google_api_key=api_key,
)

# 直接載入EMBEDDING 資料庫
db = Chroma(
    persist_directory=PERSIST_DIRECTORY, 
    embedding_function=embeddings
)

# --- 4. 設定問答鏈 ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=api_key
)

prompt = ChatPromptTemplate.from_template(
    """   
    請根據提供的診斷報告內容回答問題。
    如果報告中找不到答案，請回答「根據提供的報告內容，無法找到相關資訊」。

    <報告內容>
    {context}
    </報告內容>

    問題：{input}
    """
    )

chain = prompt | llm | StrOutputParser()

# --- 5. 進行提問 ---
query = "這份報告寫作者是誰" # 

# 檢索最相關的片段
docs = db.similarity_search(query, k=k_value)

# --- 6. 除錯資訊：印出檢索到的內容 ---
print(f"資料庫檢索回傳了 {len(docs)} 個片段")
for i, d in enumerate(docs):
    content_snippet = d.page_content.replace('\n', ' ')[:100] # 取前100字方便查看
    print(f"片段 {i+1}: {content_snippet}...")

# 執行 LLM 回答
result = chain.invoke({"input": query, "context": docs})

print(f"問：{query}")
print(f"答：\n{result}")
