from typing import List
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException
from chunk_finder import ChunkFinder

chunk_finder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер жизненного цикла приложения
    """
    # Код, выполняемый при запуске
    global chunk_finder
    try:
        # Определяем DataFrame'ы
        train_df = pd.read_csv('train_data.csv')
        chunks_df = pd.read_csv('chunks (1).csv')
        # Загружаем модель
        chunk_finder = ChunkFinder(chunks_df, train_df)
        print("Модель успешно загружена")
    except Exception as e:
        print(f"Ошибка загрузки модели: {str(e)}")
        raise
    
    yield  # Здесь работает приложение
    
    # Код, выполняемый при завершении
    chunk_finder = None
    print("Завершение работы приложения")

# Инициализация FastAPI с менеджером жизненного цикла
app = FastAPI(
    title="Chunk Finder API",
    description="API для поиска релевантных чанков по вопросу",
    lifespan=lifespan
)

@app.post("/find_chunks/")
async def find_chunks(input_data: dict):
    """
    Endpoint для поиска релевантных чанков по вопросу
    """
    if not chunk_finder:
        raise HTTPException(status_code=500, detail="Ошибка инициализации модели")
    
    question = input_data.get('question_text', '')
    
    # Получаем релевантные чанки
    results = chunk_finder.find_relevant_chunks(
        question=question,
        top_k=5
    )
    
    # Формируем ответ
    response = {
        "chunks": [result['chunk'] for result in results],
        "metadata": {
            "total_chunks": len(results),
            "question": question,
            "results_details": results
        }
    }
    
    return response

@app.post("/evaluate/")
async def evaluate_model(test_data: List[dict]):
    """
    Оценка качества модели по метрикам recall@k
    Ожидаемый формат test_data:[{"question": "текст вопроса","true_chunk": "правильный чанк"},...]
    """
    if not chunk_finder:
        raise HTTPException(status_code=500, detail="Ошибка инициализации модели")

    total = len(test_data)
    correct = {1: 0, 3: 0, 5: 0}
    
    for item in test_data:
        question = item["question"]
        true_chunk = item["true_chunk"]
        
        # Получаем предсказания
        predictions = chunk_finder.find_relevant_chunks(question, top_k=5)
        predicted_chunks = [pred['chunk'] for pred in predictions]
        
        # Проверяем recall@k
        for k in [1, 3, 5]:
            if true_chunk in predicted_chunks[:k]:
                correct[k] += 1
    
    # Вычисляем метрики
    metrics = {
        f"recall@{k}": correct[k] / total
        for k in [1, 3, 5]
    }
    
    # Формируем ответ
    response = {
        "metrics": metrics,
        "total_examples": total,
        "correct_predictions": correct
    }
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)