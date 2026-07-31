from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

class Embedder:

    def __init__(self,model_name:str=None):
        self.model_name=model_name or EMBEDDING_MODEL
        print(f"正在嵌入模型：{self.model_name}")
        print("首次加载模型可能需要一些时间，请耐心等待...")

        self.model=SentenceTransformer(self.model_name)
        print("模型加载完成！")
    def embed_texts(self,texts:list[str])->list[list[float]]:
        embeddings=self.model.encode(texts,normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        """将用户问题转为向量"""
        return self.embed_texts([query])[0]


#测试代码    
if __name__ == "__main__":
    embedder = Embedder()

    # 测试：将两段文字转为向量
    texts = [
        "RAG是一种检索增强生成技术",
        "今天天气真好啊"
    ]
    vectors = embedder.embed_texts(texts)

    print(f"文本1向量维度: {len(vectors[0])}")
    print(f"文本1向量前5位: {vectors[0][:5]}")

    # 计算两段文字的相似度（点积）
    similarity = sum(a * b for a, b in zip(vectors[0], vectors[1]))
    print(f"两段文字的相似度: {similarity:.4f}")
    print(f"(数值越接近1越相似，越接近-1越不相似)")
