import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR,TOP_K


#向量检索器
class Retriever:
    def __init__(self,embedder,collection_name:str="rag_docs"):
        self.embedder=embedder
        self.collection_name = collection_name
        self.client=chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)#关闭默认传参
        )

        self.collection=self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space":"cosine"}#使用余弦相似度
        )

        print(f"向量数据库就绪，当前文档数: {self.collection.count()}")

    def add_chunks(self,chunks:list[dict]):
        if not chunks:
            print("没有数据需要添加")
            return

        #提取块的所有内容
        texts=[chunk["content"] for chunk in chunks]
        print(f"正在向量化{len(texts)}个文本块")

        embeddings=self.embedder.embed_texts(texts)


        #准备存入数据库的数据
        ids=[f"chunk_{i}" for i in range(len(chunks))]
        metadatas=[
            {"source":chunk["source"],"chunk_id":chunk["chunk_id"]}
            for chunk in chunks
        ]
        #写入chroma数据库
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"已存入 {len(chunks)} 个文本块到向量数据库")
        print(f"数据库路径: {CHROMA_PERSIST_DIR}")
    def search(self,query:str,top_k:int=None):
        if top_k is None:
            top_k=TOP_K

       #查询问题转换为向量
        query_embedding=self.embedder.embed_query(query)

       #在向量数据库中检索



        results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]


        )
          # 3. 整理返回结果
        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                retrieved.append({
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "chunk_id": results["metadatas"][0][i]["chunk_id"],
                    # 将距离转换为相似度分数（余弦距离 -> 相似度）
                    "score": 1 - results["distances"][0][i]
                })

        return retrieved

    def clear(self):
        """清空数据库（调试用）"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("数据库已清空")


# ========== 测试代码 ==========
if __name__ == "__main__":
    from embedder import Embedder
    from loader import load_all_documents
    from chunker import TextChunker

    # 1. 加载文档
    docs = load_all_documents()

    # 2. 分块
    chunker = TextChunker()
    chunks = chunker.chunk_documents(docs)

    # 3. 初始化嵌入器和检索器
    embedder = Embedder()
    retriever = Retriever(embedder)

    # 4. 存入向量数据库
    retriever.add_chunks(chunks)

    # 5. 测试检索
    print("=" * 50)
    query = "什么是RAG?"
    print(f"搜索问题: {query}")

