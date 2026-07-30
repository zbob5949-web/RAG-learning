from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

class TextChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ",", "!", "?", "，", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        all_chunks = []  # 用来装切完块的文档
        for doc in documents:
            source = doc["source"]  # 文件名
            content = doc["content"]  # 文档正文

            chunk_texts = self.splitter.split_text(content)
            print(f"{source}: {len(content)} 字符 -> {len(chunk_texts)} 个块")

            for i, chunk_text in enumerate(chunk_texts):  # 给每个块打上索引标签
                all_chunks.append({
                    "source": source,
                    "chunk_id": i,
                    "content": chunk_text.strip()
                })

        print(f"总共生成 {len(all_chunks)} 个文本块")
        return all_chunks

# ========== 测试代码 ==========
if __name__ == "__main__":
    from loader import load_all_documents

    # 1. 加载文档
    docs = load_all_documents()

    # 2. 分块
    chunker = TextChunker()
    chunks = chunker.chunk_documents(docs)

    # 3. 查看结果
    for chunk in chunks:
        print(f"[{chunk['source']}] 块#{chunk['chunk_id']} "
              f"({len(chunk['content'])}字符):")
        print(chunk['content'][:150] + "...")
        print()
