# Copyright (c) 2026 salim-slimani — MIT License
"""pychain — بديل langchain الأسرع (نفس الأسماء، بدون pydantic/ثقل).

مثال:
    from pychain import ChatPromptTemplate, FakeListChatModel, StrOutputParser
    prompt = ChatPromptTemplate.from_template("ترجم إلى الفرنسية: {text}")
    chain = prompt | FakeListChatModel(responses=["Bonjour"]) | StrOutputParser()
    print(chain.invoke({"text": "صباح الخير"}))

مصادر السرعة:
- slots + بدون pydantic + lazy imports
- Runnable.batch متوازٍ (ThreadPool) و RunnableParallel متوازٍ
- InMemoryVectorStore متجه عبر numpy عند توفره
- TextSplitter بتمريرة واحدة و HashingEmbeddings محلي حتمي
"""
from __future__ import annotations

__version__ = "0.1.0"

from .messages import (AIMessage, BaseMessage, ChatMessage, HumanMessage,
                       SystemMessage, ToolMessage, convert_to_messages,
                       get_buffer_string)
from .documents import Document
from .prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from .splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from .embeddings import (BaseEmbeddings, FakeEmbeddings, HashingEmbeddings,
                         OpenAIEmbeddings)
from .vectorstores import Chroma, FAISS, InMemoryVectorStore, VectorStore
from .retrievers import BaseRetriever, VectorStoreRetriever
from .runnables import (Runnable, RunnableBranch, RunnableLambda,
                        RunnableParallel, RunnablePassthrough, RunnableSequence,
                        RunnableWithFallbacks, chain)
from .models import (BaseChatModel, ChatOllama, ChatOpenAI, EchoChatModel,
                     FakeChatModel, FakeListChatModel, OllamaChatModel,
                     OpenAIChatModel)
from .chains import (LLMChain, MapReduceDocumentsChain, RefineDocumentsChain,
                     RetrievalQA, SequentialChain, StuffDocumentsChain,
                     create_retrieval_chain, create_stuff_documents_chain)
from .tools import StructuredTool, Tool, tool
from .agents import AgentExecutor, create_react_agent
from .memory import (ConversationBufferMemory, ConversationBufferWindowMemory,
                     ConversationSummaryMemory)
from .parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from .cache import InMemoryCache, SQLiteCache
from .loaders import DirectoryLoader, TextLoader

__all__ = [
    "BaseMessage", "HumanMessage", "AIMessage", "SystemMessage", "ToolMessage", "ChatMessage",
    "Document", "PromptTemplate", "ChatPromptTemplate", "FewShotPromptTemplate",
    "RecursiveCharacterTextSplitter", "CharacterTextSplitter",
    "BaseEmbeddings", "FakeEmbeddings", "HashingEmbeddings", "OpenAIEmbeddings",
    "InMemoryVectorStore", "FAISS", "Chroma", "VectorStore",
    "BaseRetriever", "VectorStoreRetriever",
    "Runnable", "RunnableLambda", "RunnableSequence", "RunnableParallel",
    "RunnablePassthrough", "RunnableBranch", "RunnableWithFallbacks", "chain",
    "BaseChatModel", "FakeListChatModel", "FakeChatModel", "EchoChatModel",
    "OpenAIChatModel", "ChatOpenAI", "OllamaChatModel", "ChatOllama",
    "LLMChain", "StuffDocumentsChain", "MapReduceDocumentsChain", "RefineDocumentsChain",
    "SequentialChain", "RetrievalQA", "create_stuff_documents_chain", "create_retrieval_chain",
    "Tool", "StructuredTool", "tool", "AgentExecutor", "create_react_agent",
    "ConversationBufferMemory", "ConversationBufferWindowMemory", "ConversationSummaryMemory",
    "StrOutputParser", "JsonOutputParser", "PydanticOutputParser",
    "InMemoryCache", "SQLiteCache", "TextLoader", "DirectoryLoader",
    "convert_to_messages", "get_buffer_string",
]
