import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from tqdm import tqdm
from typing import List, Dict, Optional
import hashlib
import pickle
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vectorizer.log')
    ]
)
logger = logging.getLogger(__name__)

class DocVectorizer:
    def __init__(self, 
                 base_url: str,
                 output_dir: str = "vectors",
                 model_name: str = 'all-MiniLM-L6-v2',
                 chunk_size: int = 1000,
                 max_pages: int = 100):
        """
        初始化文档向量化器
        
        Args:
            base_url: 要爬取的网站基础URL
            output_dir: 输出目录
            model_name: 使用的向量模型名称
            chunk_size: 文本分块大小
            max_pages: 最大爬取页面数
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.max_pages = max_pages
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized DocVectorizer with base_url: {base_url}")
        
    def _get_document_id(self, url: str, text: str) -> str:
        """生成文档唯一ID"""
        return hashlib.md5(f"{url}:{text}".encode()).hexdigest()
    
    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        # 移除多余的空白字符
        text = ' '.join(text.split())
        # 移除特殊字符（可根据需要添加更多处理）
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        return text
    
    def crawl_pages(self) -> Dict[str, str]:
        """爬取网站页面"""
        visited = set()
        to_visit = {self.base_url}
        pages = {}
        
        with tqdm(total=self.max_pages, desc="Crawling pages") as pbar:
            while to_visit and len(visited) < self.max_pages:
                url = to_visit.pop()
                if url in visited or not url.startswith(self.base_url):
                    continue
                    
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 提取文本内容
                    content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'article', 'section'])
                    content = ' '.join([elem.get_text(strip=True) for elem in content_elements])
                    content = self._clean_text(content)
                    
                    if content.strip():
                        pages[url] = content
                        logger.info(f"Successfully crawled: {url}")
                    
                    # 提取新链接
                    for link in soup.find_all('a'):
                        new_url = urljoin(url, link.get('href', ''))
                        if new_url.startswith(self.base_url):
                            to_visit.add(new_url)
                            
                    visited.add(url)
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error crawling {url}: {str(e)}")
                    
        return pages
    
    def chunk_text(self, text: str) -> List[str]:
        """将文本分块"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def process_documents(self) -> List[Dict]:
        """处理文档并返回向量数据"""
        pages = self.crawl_pages()
        vectors_data = []
        
        for url, content in tqdm(pages.items(), desc="Processing documents"):
            chunks = self.chunk_text(content)
            try:
                vectors = self.model.encode(chunks, show_progress_bar=False)
                
                for chunk, vector in zip(chunks, vectors):
                    doc_id = self._get_document_id(url, chunk)
                    vectors_data.append({
                        'id': doc_id,
                        'url': url,
                        'text': chunk,
                        'vector': vector.tolist(),
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error processing document {url}: {str(e)}")
        
        return vectors_data

    def save_jsonl(self, vectors_data: List[Dict]):
        """保存为JSONL格式"""
        output_file = f"{self.output_dir}/vectors.jsonl"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in vectors_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Saved JSONL format to {output_file}")
        except Exception as e:
            logger.error(f"Error saving JSONL format: {str(e)}")

    def save_faiss(self, vectors_data: List[Dict]):
        """保存为FAISS格式"""
        try:
            import faiss
            vectors = np.array([item['vector'] for item in vectors_data])
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors.astype('float32'))
            
            faiss_dir = f"{self.output_dir}/faiss"
            os.makedirs(faiss_dir, exist_ok=True)
            
            # 保存索引
            faiss.write_index(index, f"{faiss_dir}/index.faiss")
            
            # 保存元数据
            metadata = [{k: v for k, v in item.items() if k != 'vector'} 
                       for item in vectors_data]
            with open(f"{faiss_dir}/metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"Saved FAISS format to {faiss_dir}")
        except ImportError:
            logger.warning("FAISS not installed. Skip saving FAISS format.")
        except Exception as e:
            logger.error(f"Error saving FAISS format: {str(e)}")

    def save_chroma(self, vectors_data: List[Dict]):
        """保存为Chroma格式"""
        try:
            import chromadb
            
            client = chromadb.Client()
            collection_name = "doc_vectors"
            
            # 如果集合已存在，先删除
            try:
                client.delete_collection(collection_name)
            except:
                pass
                
            collection = client.create_collection(name=collection_name)
            
            # 插入数据
            collection.add(
                ids=[item['id'] for item in vectors_data],
                embeddings=[item['vector'] for item in vectors_data],
                metadatas=[{
                    'url': item['url'],
                    'timestamp': item['timestamp']
                } for item in vectors_data],
                documents=[item['text'] for item in vectors_data]
            )
            
            logger.info(f"Saved to Chroma collection: {collection_name}")
            
        except ImportError:
            logger.warning("Chromadb not installed. Skip saving Chroma format.")
        except Exception as e:
            logger.error(f"Error saving Chroma format: {str(e)}")

    def save_langchain(self, vectors_data: List[Dict]):
        """保存为LangChain格式"""
        try:
            from langchain.docstore.document import Document
            from langchain.vectorstores import FAISS as LangchainFAISS
            
            documents = []
            for item in vectors_data:
                doc = Document(
                    page_content=item['text'],
                    metadata={
                        'source': item['url'],
                        'id': item['id'],
                        'timestamp': item['timestamp']
                    }
                )
                documents.append(doc)
            
            # 创建向量存储
            vectorstore = LangchainFAISS.from_documents(documents, self.model)
            # 保存
            langchain_dir = f"{self.output_dir}/langchain"
            vectorstore.save_local(langchain_dir)
            logger.info(f"Saved LangChain format to {langchain_dir}")
            
        except ImportError:
            logger.warning("Langchain not installed. Skip saving LangChain format.")
        except Exception as e:
            logger.error(f"Error saving LangChain format: {str(e)}")

    def process_and_save_all(self, save_formats: List[str] = None):
        """
        处理文档并保存指定格式
        
        Args:
            save_formats: 要保存的格式列表，可选值：['jsonl', 'faiss', 'chroma', 'langchain']
                        如果为None，则保存所有支持的格式
        """
        vectors_data = self.process_documents()
        
        if not vectors_data:
            logger.warning("No documents processed. Skipping save operations.")
            return
            
        if save_formats is None:
            save_formats = ['jsonl', 'faiss', 'chroma', 'langchain']
            
        for format_name in save_formats:
            try:
                if format_name == 'jsonl':
                    self.save_jsonl(vectors_data)
                elif format_name == 'faiss':
                    self.save_faiss(vectors_data)
                elif format_name == 'chroma':
                    self.save_chroma(vectors_data)
                elif format_name == 'langchain':
                    self.save_langchain(vectors_data)
                else:
                    logger.warning(f"Unknown format: {format_name}")
            except Exception as e:
                logger.error(f"Error saving {format_name} format: {str(e)}")

def main():
    # 配置参数
    config = {
        'base_url': 'https://docs.solanatracker.io',  # 替换为实际的网站URL
        'output_dir': 'vector_data',
        'model_name': 'all-MiniLM-L6-v2',
        'chunk_size': 1000,
        'max_pages': 100
    }
    
    try:
        # 初始化向量化器
        vectorizer = DocVectorizer(**config)
        
        # 处理并保存所有格式
        # 可以指定特定格式：['jsonl', 'faiss', 'chroma', 'langchain']
        vectorizer.process_and_save_all(['jsonl', 'faiss', 'langchain'])
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()