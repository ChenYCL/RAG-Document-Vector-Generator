````markdown
# RAG Document Vector Generator

一个用于生成和测试文档向量的 RAG (Retrieval-Augmented Generation) 工具集。

## 功能特点

- 支持网页文档爬取和处理
- 多种向量存储格式支持：
  - JSONL
  - FAISS
  - Chroma
  - LangChain
- 智能文本分块
- 向量相似度搜索
- 支持批量测试和交互式测试

## 安装

1. 创建虚拟环境：

```bash
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# 或
.\myenv\Scripts\activate  # Windows
```
````

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 生成向量数据

```python
from main import DocVectorizer

vectorizer = DocVectorizer(
    base_url="https://your-docs-site.com",
    output_dir="vector_data",
    chunk_size=1000
)
vectorizer.process_and_save_all()
```

### 2. 测试向量检索

```python
python test_rag.py
```

## 配置参数

- `base_url`: 要爬取的文档网站基础 URL
- `output_dir`: 向量数据输出目录
- `chunk_size`: 文本分块大小
- `model_name`: 向量模型名称（默认：'all-MiniLM-L6-v2'）
- `max_pages`: 最大爬取页面数

## 向量存储格式

1. JSONL 格式

   - 位置：`{output_dir}/vectors.jsonl`
   - 用途：通用格式，易于处理和迁移

2. FAISS 格式

   - 位置：`{output_dir}/faiss/`
   - 用途：高效的向量检索

3. Chroma 格式

   - 位置：Chroma 数据库
   - 用途：简单易用的向量存储

4. LangChain 格式
   - 位置：`{output_dir}/langchain/`
   - 用途：与 LangChain 生态系统集成

## 开发

### 运行测试

```bash
python test_rag.py
```

### 添加新的向量存储格式

1. 在 `DocVectorizer` 类中添加新的保存方法
2. 在 `RAGTester` 类中添加相应的测试方法
3. 更新 `process_and_save_all` 方法

## 项目结构

```
rag-generate/
├── main.py              # 向量生成主程序
├── test_rag.py         # 向量检索测试程序
├── requirements.txt    # 依赖包列表
├── README.md          # 项目文档
└── vector_data/       # 生成的向量数据
    ├── vectors.jsonl
    ├── faiss/
    └── langchain/
```

## 依赖项

- sentence-transformers
- requests
- beautifulsoup4
- tqdm
- faiss-cpu
- chromadb
- langchain
- langchain-community
- numpy

## 注意事项

- 确保有足够的磁盘空间存储向量数据
- 对于大型文档集，建议适当调整 chunk_size
- 注意网站的爬虫策略和限制

## License

MIT
