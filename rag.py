import os 
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex #SimpleDirectoryReader


# Place your document
loader = PyMuPDFReader()
documents = loader.load(file_path = "/Users/harry/Desktop/rag_llm/Operation system 10th edition.pdf")
os.environ["TOKENIZERS_PARALLELISM"] = "false" # turn down warning msg

# Place your api key 
# os.environ["OPENAI_API_KEY"] = "sk-X"
# access_token = "hf_X"

# 放 text2vec 模型
# BAAI/bge-small-zh
# BAAI/bge-small-en
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

query_engine = index.as_query_engine()
response = query_engine.query("what is multicore programming?")

print(response)
print("\n\n")

retriever = index.as_retriever()
nodes = retriever.retrieve("what is multicore programming")

def print_formatted_nodes(nodes):
    for node_with_score in nodes:
        node = node_with_score.node
        print(node.text)
        print("Used pdf:")
        for key, value in node.metadata.items():
            print(f"  {key}: {value}")
        print("Score:", node_with_score.score)
        print("-" * 80)  # 打印分隔线

print_formatted_nodes(nodes)