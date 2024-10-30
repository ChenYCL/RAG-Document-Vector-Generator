import faiss
import numpy as np
import pickle
import json
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS as LangchainFAISS
import chromadb
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class RAGTester:
    def __init__(self, vector_data_dir: str = "vector_data"):
        self.vector_data_dir = vector_data_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def encode_query(self, query: str) -> np.ndarray:
        """将查询文本转换为向量"""
        return self.model.encode([query])[0]

    def test_jsonl(self, query: str, top_k: int = 3) -> List[Dict]:
        """测试JSONL格式的向量检索"""
        try:
            # 加载JSONL数据
            vectors_data = []
            with open(f"{self.vector_data_dir}/vectors.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    vectors_data.append(json.loads(line))
            
            # 将查询转换为向量
            query_vector = self.encode_query(query)
            
            # 计算相似度并排序
            results = []
            for item in vectors_data:
                similarity = np.dot(query_vector, item['vector']) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(item['vector'])
                )
                results.append({
                    'text': item['text'],
                    'url': item['url'],
                    'similarity': float(similarity)
                })
            
            # 按相似度排序
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error testing JSONL format: {e}")
            return []

    def test_faiss(self, query: str, top_k: int = 3) -> List[Dict]:
        """测试FAISS格式的向量检索"""
        try:
            # 加载FAISS索引
            index = faiss.read_index(f"{self.vector_data_dir}/faiss/index.faiss")
            
            # 加载元数据
            with open(f"{self.vector_data_dir}/faiss/metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            # 将查询转换为向量
            query_vector = self.encode_query(query)
            
            # 搜索最相似的向量
            D, I = index.search(query_vector.reshape(1, -1).astype('float32'), top_k)
            
            # 组织结果
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < len(metadata):
                    results.append({
                        'text': metadata[idx]['text'],
                        'url': metadata[idx]['url'],
                        'similarity': float(score)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing FAISS format: {e}")
            return []

    def test_langchain(self, query: str, top_k: int = 3) -> List[Dict]:
        """测试LangChain格式的向量检索"""
        try:
            # 加载向量存储
            vectorstore = LangchainFAISS.load_local(
                f"{self.vector_data_dir}/langchain",
                self.model,
                allow_dangerous_deserialization=True  # 添加这个参数
            )
            
            # 搜索相似文档
            docs = vectorstore.similarity_search_with_score(query, k=top_k)
            
            # 组织结果
            results = []
            for doc, score in docs:
                results.append({
                    'text': doc.page_content,
                    'url': doc.metadata.get('source', ''),
                    'similarity': float(score)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing LangChain format: {e}")
            return []

    def test_chroma(self, query: str, top_k: int = 3) -> List[Dict]:
        """测试Chroma格式的向量检索"""
        try:
            # 初始化Chroma客户端
            client = chromadb.Client()
            collection = client.get_collection("doc_vectors")
            
            # 查询
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # 组织结果
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'url': results['metadatas'][0][i]['url'],
                    'similarity': float(results['distances'][0][i])
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error testing Chroma format: {e}")
            return []

    def run_all_tests(self, query: str, top_k: int = 3):
        """运行所有格式的测试"""
        logger.info(f"Testing with query: {query}")
        
        # 测试所有格式
        test_functions = {
            'JSONL': self.test_jsonl,
            'FAISS': self.test_faiss,
            'LangChain': self.test_langchain,
            'Chroma': self.test_chroma
        }
        
        results = {}
        for format_name, test_func in test_functions.items():
            logger.info(f"\nTesting {format_name} format:")
            try:
                format_results = test_func(query, top_k)
                results[format_name] = format_results
                
                # 打印结果
                if format_results:
                    for i, result in enumerate(format_results, 1):
                        logger.info(f"\nResult {i}:")
                        logger.info(f"Text: {result['text'][:200]}...")
                        logger.info(f"URL: {result['url']}")
                        logger.info(f"Similarity: {result['similarity']:.4f}")
                else:
                    logger.info("No results found")
                    
            except Exception as e:
                logger.error(f"Error in {format_name} test: {e}")
                
        return results

def main():
    # 示例查询
    test_queries = [
        "api",
        "swap",
        "buy"
    ]
    
    # 初始化测试器
    tester = RAGTester(vector_data_dir="vector_data")
    
    # 运行测试
    for query in test_queries:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing query: {query}")
        logger.info(f"{'='*50}")
        
        results = tester.run_all_tests(query, top_k=3)

if __name__ == "__main__":
    main()