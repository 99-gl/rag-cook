"""
RAG系统Streamlit应用
"""

import os
import sys
import streamlit as st
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 页面配置
st.set_page_config(
    page_title="尝尝咸淡RAG系统",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 10px;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        align-self: flex-end;
    }
    .assistant-message {
        background-color: #f1f8e9;
        align-self: flex-start;
    }
</style>
""", unsafe_allow_html=True)

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("MODELscope_API_KEY"):
            raise ValueError("请设置 MODELscope_API_KEY 环境变量")
    
    def initialize_system(self):
        """初始化所有模块"""
        # 1. 初始化数据准备模块
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
    
    def build_knowledge_base(self):
        """构建知识库"""
        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            # 仍需要加载文档和分块用于检索模块
            self.data_module.load_documents()
            chunks = self.data_module.chunk_documents()
        else:
            # 2. 加载文档
            self.data_module.load_documents()

            # 3. 文本分块
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 返回统计信息
        stats = self.data_module.get_statistics()
        return stats
    
    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        
        # 1. 查询路由
        route_type = self.generation_module.query_router(question)

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            # 列表查询保持原查询
            rewritten_query = question
        else:
            # 详细查询和一般查询使用智能重写
            rewritten_query = self.generation_module.query_rewrite(question)
        
        # 3. 检索相关子块（自动应用元数据过滤）
        filters = self._extract_filters_from_query(question)
        if filters:
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=self.config.top_k)
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # 5. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回菜品名称列表
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            return self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            # 详细查询：获取完整文档并生成详细回答
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 根据路由类型自动选择回答模式
            if route_type == "detail":
                # 详细查询使用分步指导模式
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_step_by_step_answer(question, relevant_docs)
            else:
                # 一般查询使用基础回答模式
                if stream:
                    return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_basic_answer(question, relevant_docs)
    
    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters

# 初始化应用
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.initialized = False
    st.session_state.knowledge_base_built = False
    st.session_state.chat_history = []

# 侧边栏
with st.sidebar:
    st.title("🍽️ 尝尝咸淡RAG系统")
    st.markdown("解决您的选择困难症，告别'今天吃什么'的世纪难题！")
    
    if not st.session_state.initialized:
        if st.button("初始化系统"):
            try:
                st.session_state.rag_system = RecipeRAGSystem()
                st.session_state.rag_system.initialize_system()
                st.session_state.initialized = True
                st.success("系统初始化成功！")
            except Exception as e:
                st.error(f"初始化失败: {e}")
    
    if st.session_state.initialized and not st.session_state.knowledge_base_built:
        if st.button("构建知识库"):
            try:
                with st.spinner("正在构建知识库..."):
                    stats = st.session_state.rag_system.build_knowledge_base()
                    st.session_state.knowledge_base_built = True
                    st.success("知识库构建成功！")
                    # 显示统计信息
                    st.markdown("### 知识库统计")
                    st.markdown(f"- 文档总数: {stats['total_documents']}")
                    st.markdown(f"- 文本块数: {stats['total_chunks']}")
                    st.markdown(f"- 菜品分类: {list(stats['categories'].keys())}")
                    st.markdown(f"- 难度分布: {stats['difficulties']}")
            except Exception as e:
                st.error(f"构建知识库失败: {e}")

# 主界面
st.title("🍽️ 尝尝咸淡RAG系统")

# 聊天界面
if st.session_state.knowledge_base_built:
    # 显示聊天历史
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
    
    # 输入框
    user_input = st.chat_input("请输入您的问题...")
    
    if user_input:
        # 添加用户消息到历史
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 生成回答
        with st.chat_message("assistant"):
            # 流式输出
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                for chunk in st.session_state.rag_system.ask_question(user_input, stream=True):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                
                # 添加助手消息到历史
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_message = f"处理问题时出错: {e}"
                response_placeholder.markdown(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
else:
    st.info("请先在侧边栏初始化系统并构建知识库")
