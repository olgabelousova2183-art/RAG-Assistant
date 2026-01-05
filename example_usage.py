"""
Пример использования RAG ассистента
"""

from rag_assistant import RAGAssistant
import os

def example_basic_usage():
    """Базовый пример использования"""
    print("=" * 60)
    print("Пример 1: Базовое использование")
    print("=" * 60)
    
    # Инициализация
    assistant = RAGAssistant()
    
    # Загрузка документов
    pdf_path = "База_знаний_НейроСфера.pdf"
    if os.path.exists(pdf_path):
        documents = assistant.load_documents([pdf_path])
        
        # Создание векторного хранилища
        assistant.create_vectorstore(documents)
        
        # Настройка цепочки вопрос-ответ
        assistant.setup_qa_chain(k=4)
        
        # Задать вопрос
        result = assistant.ask("Что такое нейросфера?")
        print("\n" + "=" * 60)
    else:
        print(f"⚠️  Файл {pdf_path} не найден")


def example_with_existing_db():
    """Пример использования существующей БД"""
    print("=" * 60)
    print("Пример 2: Использование существующей векторной БД")
    print("=" * 60)
    
    assistant = RAGAssistant()
    
    # Загрузка существующего хранилища
    if os.path.exists("./chroma_db"):
        assistant.load_existing_vectorstore()
        assistant.setup_qa_chain()
        
        # Задать несколько вопросов
        questions = [
            "Какие основные темы в документе?",
            "Что можно узнать из базы знаний?",
        ]
        
        for question in questions:
            assistant.ask(question)
            print("\n" + "-" * 60 + "\n")
    else:
        print("⚠️  Векторное хранилище не найдено. Сначала создайте его.")


def example_custom_settings():
    """Пример с кастомными настройками"""
    print("=" * 60)
    print("Пример 3: Кастомные настройки")
    print("=" * 60)
    
    assistant = RAGAssistant(
        model_name="gpt-3.5-turbo",  # Более быстрая модель
        embedding_model="text-embedding-3-small",
    )
    
    pdf_path = "База_знаний_НейроСфера.pdf"
    if os.path.exists(pdf_path):
        documents = assistant.load_documents([pdf_path])
        
        # Меньшие чанки для более точного поиска
        assistant.create_vectorstore(
            documents,
            chunk_size=500,
            chunk_overlap=100
        )
        
        # Больше документов для контекста
        assistant.setup_qa_chain(k=6)
        
        result = assistant.ask("Расскажи подробно о содержимом документа")
        print("\n" + "=" * 60)


if __name__ == "__main__":
    # Выберите пример для запуска
    print("Выберите пример для запуска:")
    print("1. Базовое использование")
    print("2. Использование существующей БД")
    print("3. Кастомные настройки")
    
    choice = input("\nВведите номер (1-3): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_with_existing_db()
    elif choice == "3":
        example_custom_settings()
    else:
        print("Неверный выбор. Запускаю пример 1...")
        example_basic_usage()

