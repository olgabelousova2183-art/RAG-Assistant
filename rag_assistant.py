"""
RAG Assistant - Retrieval Augmented Generation Assistant
Подключается к OpenAI API для создания умного ассистента с базой знаний
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Optional

# Загружаем переменные окружения
load_dotenv()


class RAGAssistant:
    """RAG ассистент с подключением к OpenAI API"""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4-turbo-preview",
        embedding_model: str = "text-embedding-3-small",
        persist_directory: str = "./chroma_db"
    ):
        """
        Инициализация RAG ассистента
        
        Args:
            openai_api_key: API ключ OpenAI (если не указан, берется из .env)
            model_name: Название модели для генерации ответов
            embedding_model: Название модели для embeddings
            persist_directory: Директория для сохранения векторной БД
        """
        # Получаем API ключ
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY не найден! "
                "Укажите его в .env файле или передайте как параметр."
            )
        
        # Инициализируем модели
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=self.api_key
        )
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            openai_api_key=self.api_key
        )
        
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        
        print("✅ RAG ассистент инициализирован")
    
    def load_documents(self, file_paths: List[str]) -> List:
        """
        Загружает документы из файлов
        
        Args:
            file_paths: Список путей к файлам (PDF, TXT и т.д.)
        
        Returns:
            Список загруженных документов
        """
        documents = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"⚠️  Файл не найден: {file_path}")
                continue
            
            print(f"📄 Загрузка документа: {file_path}")
            
            # Поддержка PDF файлов
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"   Загружено {len(docs)} страниц")
            else:
                print(f"   ⚠️  Формат файла не поддерживается: {file_path}")
        
        print(f"✅ Всего загружено документов: {len(documents)}")
        return documents
    
    def create_vectorstore(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Создает векторное хранилище из документов
        
        Args:
            documents: Список документов
            chunk_size: Размер чанков текста
            chunk_overlap: Перекрытие между чанками
        """
        if not documents:
            raise ValueError("Нет документов для обработки!")
        
        print("🔨 Разбиение документов на чанки...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Создано {len(chunks)} чанков")
        
        print("🔨 Создание векторного хранилища...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✅ Векторное хранилище создано и сохранено в {self.persist_directory}")
    
    def load_existing_vectorstore(self):
        """Загружает существующее векторное хранилище"""
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"Векторное хранилище не найдено в {self.persist_directory}")
        
        print(f"📂 Загрузка векторного хранилища из {self.persist_directory}...")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print("✅ Векторное хранилище загружено")
    
    def setup_qa_chain(self, k: int = 4):
        """
        Настраивает цепочку вопрос-ответ
        
        Args:
            k: Количество релевантных документов для извлечения
        """
        if not self.vectorstore:
            raise ValueError("Векторное хранилище не создано! Сначала загрузите документы.")
        
        # Создаем кастомный промпт
        prompt_template = """Используй следующие фрагменты контекста из базы знаний, чтобы ответить на вопрос.
Если ты не знаешь ответа, просто скажи, что не знаешь, не пытайся придумать ответ.

Контекст: {context}

Вопрос: {question}

Дай подробный и точный ответ на основе предоставленного контекста:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Создаем retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Создаем QA цепочку
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("✅ Цепочка вопрос-ответ настроена")
    
    def ask(self, question: str) -> dict:
        """
        Задает вопрос ассистенту
        
        Args:
            question: Вопрос пользователя
        
        Returns:
            Словарь с ответом и исходными документами
        """
        if not self.qa_chain:
            raise ValueError("Цепочка вопрос-ответ не настроена! Вызовите setup_qa_chain()")
        
        print(f"\n❓ Вопрос: {question}")
        print("🔍 Поиск релевантной информации...")
        
        result = self.qa_chain.invoke({"query": question})
        
        print(f"\n💬 Ответ:\n{result['result']}")
        print(f"\n📚 Использовано источников: {len(result.get('source_documents', []))}")
        
        return result
    
    def interactive_mode(self):
        """Запускает интерактивный режим для общения с ассистентом"""
        if not self.qa_chain:
            print("⚠️  Сначала настройте цепочку вопрос-ответ!")
            return
        
        print("\n" + "="*60)
        print("🤖 RAG Ассистент готов к работе!")
        print("Введите 'выход' или 'exit' для завершения")
        print("="*60 + "\n")
        
        while True:
            question = input("Ваш вопрос: ").strip()
            
            if question.lower() in ['выход', 'exit', 'quit', 'q']:
                print("👋 До свидания!")
                break
            
            if not question:
                continue
            
            try:
                self.ask(question)
                print("\n" + "-"*60 + "\n")
            except Exception as e:
                print(f"❌ Ошибка: {e}\n")


def main():
    """Основная функция для запуска RAG ассистента"""
    
    # Инициализация ассистента
    assistant = RAGAssistant()
    
    # Путь к PDF файлу
    pdf_path = "База_знаний_НейроСфера.pdf"
    
    # Проверяем, существует ли уже векторное хранилище
    if os.path.exists("./chroma_db"):
        print("📂 Найдено существующее векторное хранилище")
        use_existing = input("Использовать существующее хранилище? (y/n): ").strip().lower()
        
        if use_existing == 'y':
            assistant.load_existing_vectorstore()
        else:
            # Загружаем документы и создаем новое хранилище
            documents = assistant.load_documents([pdf_path])
            assistant.create_vectorstore(documents)
    else:
        # Загружаем документы и создаем новое хранилище
        documents = assistant.load_documents([pdf_path])
        assistant.create_vectorstore(documents)
    
    # Настраиваем цепочку вопрос-ответ
    assistant.setup_qa_chain(k=4)
    
    # Запускаем интерактивный режим
    assistant.interactive_mode()


if __name__ == "__main__":
    main()

