#创建主流程

from  src.embedder import Embedder
from  src.loader import load_all_documents
from  src.generator import Generator
from  src.chunker import TextChunker
from  src.retriever import Retriever

class RAGPipeline:
    def __init__(self,rebuild:bool=False):
        print("="*60)
        print("正在初始化RAG系统")
        print("="*60)

        #初始化嵌入器
        self.embedder=Embedder()
        #初始化检索器
        self.retriever=Retriever(self.embedder)
       #数据库为空则要求重建
        if rebuild or self.retriever.collection.count() ==0:
            self._build_knowledge_base()


       #初始化生成器
        self.generator=Generator()
        print("=" * 60)
        print("RAG 系统就绪！可以开始提问")
        print("=" * 60)

    def _build_knowledge_base(self):
        print("===开始构建知识库===")

#加载文档
        documents=load_all_documents()
        if not documents:
            print("请先放入文档 当前支持的文档格式为txt,pdf,docx")
            return


        #文本分块
        chunker=TextChunker()
        chunks=chunker.chunk_documents(documents)


        #清空旧的数据

        if self.retriever.collection.count() > 0:
            self.retriever.clear()

        #向数据库存入数据
        self.retriever.add_chunks(chunks)

        print("===知识库构建完成===")


    def ask(self,query:str,top_k:int=3):
        #检索相关文档
        results=self.retriever.search(query,top_k=top_k)
        print(f"检索到{len(results)}条相关文档")

        #基于检索结果生成回答
        answer=self.generator.generate(query,results)
        return{
                "query":query,
                "answer":answer,
                "sources":[{"source":r["source"],"score":r["score"]} for r in results]
        }
        #全局单例

rag=None

def get_rag(rebuild:bool=False):
    global rag
    if rag is None:
        rag=RAGPipeline(rebuild=rebuild)
    return rag
