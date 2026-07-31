from openai import OpenAI
from src.config import DEEPSEEK_API_KEY,LLM_URL,LLM_MODEL

class Generator:
    def __init__(self,api_key:str = None,model:str = None,url:str = None):
        self.api_key=api_key or DEEPSEEK_API_KEY
        self.model=model or LLM_MODEL
        self.url=url or LLM_URL


        #初始化客户端
        self.client=OpenAI(
            api_key=self.api_key,
            base_url=self.url
        )
        print(f"生成器就绪,模型：{self.model}")


    def generate(self,query:str,contexts:list[dict]):
        context_text=""

        for i, ctx in enumerate(contexts):
            context_text += f"\n参考资料来源:{i+1} (来源：{ctx['source']})"
            context_text += ctx['content'] + "\n"


           
           
        system_prompt = """你是一个基于参考资料的问答助手。
请遵循以下规则：
1. 只根据下方提供的【参考资料】来回答问题
2. 如果参考资料中没有相关信息，请如实说
"参考资料中未包含相关信息"
3. 回答时尽量引用参考资料中的具体内容
4. 保持回答简洁、准确"""

        user_prompt = f"""【参考资料】
{context_text}             

【用户问题】
{query}
请基于以上参考资料回答用户问题。"""

        print(f"正在调用{self.model}生成回答...")
        response=self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000

        )

#提取回答文本

        answer=response.choices[0].message.content

        usage=response.usage
        print(f"Token用量: 输入{usage.prompt_tokens} + 输出{usage.completion_tokens} = {usage.total_tokens}")



        return answer

if __name__ == "__main__":

    from embedder import Embedder
    from retriever import Retriever

    # 初始化各模块
    embedder = Embedder()
    retriever = Retriever(embedder)
    generator = Generator()

    # 测试问答
    query = "RAG有什么优势?"
    print(f"用户问题: {query}")

    # 检索
    results = retriever.search(query, top_k=3)
    print(f"检索到 {len(results)} 条相关文档")

    # 生成
    answer = generator.generate(query, results)
    print(f"模型回答:\n{answer}")



