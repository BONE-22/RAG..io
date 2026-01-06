from configparser import ConfigParser #讀取與管理設定檔
from langchain_community.document_loaders import Docx2txtLoader #讀取檔案(word)
from langchain_text_splitters import CharacterTextSplitter #切分文字
from langchain_google_genai import GoogleGenerativeAIEmbeddings #Google提供Embedding模型
from langchain_community.vectorstores import Chroma #向量資料庫

# --- 1. 設定區 ---
FILE_PATH = "貿特198診斷報告Final.docx"
EMBED_MODEL = "models/text-embedding-004" 
PERSIST_DIRECTORY = "./chroma_db_medical"

# --- 2. 載入金鑰 ---
config = ConfigParser()
config.read("config.ini")
api_key = config["Gemini"]["API_KEY"]

# --- 3. 處理文件 ---
print(f"正在載入檔案: {FILE_PATH}...")
loader = Docx2txtLoader(FILE_PATH)
data = loader.load()

# 切割文字
text_splitter = CharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=100,
    separator="\n"
)
docs = text_splitter.split_documents(data)

# --- 4. 建立 Embedding 與儲存資料庫 ---
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    google_api_key=api_key,
)

print(f"正在建立資料庫並儲存至 {PERSIST_DIRECTORY}...")

# 建立 Chroma 資料庫並持久化儲存
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

print("--- 資料庫建立完成！ ---")
