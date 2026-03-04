构建索引花了5，6分钟

原始数据已经是层次化的，设计为父子文档，父文档为原始文档，子文档为原始文档的切分，子文档做为chunk构建生成索引，检索后将对应父文档传给llm

数据模块：原始文档为父文档 documents，附加元数据（source、parent_id、doc_type、category、dish_name、difficulty）；对 documents 按 markdown 标题切分为子文档 chunks，附加元数据

索引构建模块：
- 初始化嵌入模型（默认使用 BAAI/bge-small-zh-v1.5），设置为 CPU 运行
- 构建 FAISS 向量索引，将文档块转换为向量并存储
- 支持向现有索引添加新文档，实现增量更新
- 提供索引保存和加载功能，避免重复构建
- 实现相似度搜索，返回与查询最相关的文档

检索优化模块：
- 结合向量检索和 BM25 检索进行混合搜索
- 使用 RRF (Reciprocal Rank Fusion) 算法重排搜索结果，提高检索准确性
- 支持基于元数据的过滤搜索，如按分类、难度等筛选

生成模块：

