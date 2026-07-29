import os
from pathlib import Path
from config import DOCS_DIR


def load_single_file(file_path:str)->str:
    #根据文件类型加载单个文件
    #获取扩展名并转小写
    ext=Path(file_path).suffix.lower()
    if ext==".txt":
        return load_txt(file_path)
    elif ext==".pdf":
        return load_pdf(file_path)
    elif ext==".docx":
        return load_docx(file_path)
    else:
        print(f"不支持的文件格式 :{ext}")
        return ""
def load_txt(file_path:str)->str:
    with open(file_path,"r",encoding="utf-8")as f:
        return f.read()

def load_pdf(file_path:str)->str:
    from pypdf import PdfReader
    reader=PdfReader(file_path)
    text_parts=[]
    for page in reader.pages:
        page_text=page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)
def load_docx(file_path:str)->str:
    from docx import Document
    doc = Document(file_path)
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    return "\n".join(text_parts)


#加载指定目录下的所有文件
def load_all_documents(docs_dir:str=None)->list[dict]:
    if docs_dir is None:
        docs_dir=DOCS_DIR
    documents=[]
    supports_exts={".txt",".pdf",".docx"}

    #遍历目录下所有文件
    for root,dirs,files in os.walk(docs_dir):
        for filename in files:
            file_path=os.path.join(root,filename)
            ext=Path(file_path).suffix.lower()
            if ext  in supports_exts:
                print(f"正在加载：{filename}")
                content=load_single_file(file_path)
                if content:
                    documents.append({"source":filename,"content":content})
    print(f"加载成功！共{len(documents)}个文档")
    return documents

if __name__ == "__main__":
    docs = load_all_documents()
    for doc in docs:
        print(f"文档: {doc['source']}")
        print(f"内容预览: {doc['content'][:200]}...")

