from typing import List, Dict
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymystem3 import Mystem

# Загрузим стоп слова для обработки русской речи
nltk.download('stopwords')
nltk.download('punkt_tab')
russian_stopwords = stopwords.words('russian')
mystem = Mystem()

# Ф-ция для стандартной очистки слов в тексте
def preprocess_text_tfidf(text: str) -> str:
    '''Предобработка текста для TF-IDF, включая приведение к нижнему регистру, удаление спец. символов, токенизацию, удаление стоп-слов и лемматизацию'''
    # Приведем к нижнему регистру
    text = text.lower()
    # Удалим спец. символы
    text = re.sub(r'[^\w\s]', ' ', text)
    # Токенизация
    tokens = word_tokenize(text)
    # Удаляем стоп-слова
    tokens = [token for token in tokens if token not in russian_stopwords]
    # Лемматизация
    lemmatized_tokens = mystem.lemmatize(' '.join(tokens))
    # Удаляем пробелы и пустые строки после лемматизации
    lemmatized_tokens = [token.strip() for token in lemmatized_tokens if token.strip()]
    return ' '.join(lemmatized_tokens)


class ChunkFinder:
    def __init__(self, chunks_df, train_df, bert_similarity_threshold=0.8):
        '''
        Класс для поиска релевантных чанков по вопросу пользователя

        Аргументы:
            chunks_df - DataFrame с чанками, содержащий колонки 'Headers' и 'Chunk'
            train_df - DataFrame с тренировочными вопросами, содержащий колонки 'Question' и 'ChunkId'
            bert_similarity_threshold - пороговое значение для схожести векторов BERT, выше которого чанк считается релевантным
        '''
        self.chunks_df = chunks_df
        self.train_df = train_df
        self.bert_similarity_threshold = bert_similarity_threshold
        
        # Инициализация моделей
        self.bert_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.tfidf = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))
        
        # Подготовка данных
        self._prepare_data()

    def _prepare_data(self):
        # Подготовка chunks_df, объединяем заголовки и текст чанков и применяем предобработку
        self.chunks_df['full_text'] = self.chunks_df['Headers'] + ' ' + self.chunks_df['Chunk']
        self.chunks_df['processed_text'] = self.chunks_df['full_text'].apply(preprocess_text_tfidf)
        
        # Подготовка train_df
        self.train_df['full_text'] = self.train_df['Headers'] + ' ' + self.train_df['Chunk']
        self.train_df['processed_text'] = self.train_df['Question'].apply(preprocess_text_tfidf)
        
        # Создание эмбеддингов для двух методов - TF-IDF и BERT для чанков
        self.chunks_embeddings_tfidf = self.tfidf.fit_transform(self.chunks_df['processed_text'])
        self.chunks_embeddings_bert = self.bert_model.encode(self.chunks_df['full_text'].tolist(), show_progress_bar=True)
        
        # Создание эмбеддингов для тренировочных вопросов
        self.train_questions_bert = self.bert_model.encode(self.train_df['Question'].tolist(), show_progress_bar=True)

    def _find_similar_training_question(self, question: str) -> tuple:
        """
        Поиск похожего вопроса в тренировочной выборке

        Аргументы:
            question: текст вопроса
            
        Возвращает:
            tuple: (индекс найденного вопроса, значение схожести) или (None, 0)
        """
        # Получаем эмбеддинг для нового вопроса
        question_embedding = self.bert_model.encode([question])
        
        # Вычисляем схожесть с тренировочными вопросами
        similarities = cosine_similarity(question_embedding, self.train_questions_bert)[0]
        
        # Находим максимальную схожесть и соответствующий индекс
        max_similarity = np.max(similarities)
        max_similarity_idx = np.argmax(similarities)
        
        # Если схожесть выше порога, возвращаем результат
        if max_similarity >= self.bert_similarity_threshold:
            return max_similarity_idx, max_similarity
            
        return None, max_similarity
    
    def find_relevant_chunks(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Поиск релевантных чанков с учетом тренировочной выборки
        
        Аргументы:
            question: текст вопроса
            top_k: количество возвращаемых результатов
            
        Возвращает:
            List[Dict]: список найденных чанков с метаданными
        """
        # Сначала ищем похожий вопрос в тренировочной выборке
        similar_idx, similarity_score = self._find_similar_training_question(question)
        
        if similar_idx is not None:
            # Если нашли похожий вопрос, возвращаем соответствующий чанк
            train_row = self.train_df.iloc[similar_idx]
            return [{
                'chunk': train_row['Chunk'],
                'header': train_row['Headers'],
                'similarity_score': float(similarity_score),
                'match_type': 'training_match',
                'original_question': train_row['Question'],
                'source': 'training_set'
            }]
        
        # Если не нашли похожий вопрос, используем комбинированный поиск
        # BERT similarities
        question_bert_emb = self.bert_model.encode([question])
        bert_similarities = cosine_similarity(question_bert_emb, self.chunks_embeddings_bert).flatten()
        
        # TF-IDF similarities
        processed_question = preprocess_text_tfidf(question)
        question_tfidf = self.tfidf.transform([processed_question])
        tfidf_similarities = cosine_similarity(question_tfidf, self.chunks_embeddings_tfidf).flatten()
        
        # Комбинируем scores (можно настроить веса)
        combined_similarities = (bert_similarities + tfidf_similarities)
        
        # Получаем топ результаты
        top_indices = combined_similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks_df.iloc[idx]['Chunk'],
                'header': self.chunks_df.iloc[idx]['Headers'],
                'similarity_score': float(combined_similarities[idx]),
                'bert_score': float(bert_similarities[idx]),
                'tfidf_score': float(tfidf_similarities[idx]),
                'match_type': 'semantic_search',
                'source': 'chunks_database'
            })
        
        return results
    
    def evaluate(self, test_questions: List[str], true_chunks: List[str], ks: List[int] = [1, 3, 5]) -> Dict:
        """
        Оценка качества модели
        
        Аргументы:
            test_questions: список тестовых вопросов
            true_chunks: список правильных чанков
            ks: список значений k для подсчета recall@k
            
        Возвращает:
            Dict: метрики качества
        """
        metrics = {f'recall@{k}': 0.0 for k in ks}
        total = len(test_questions)
        
        for question, true_chunk in zip(test_questions, true_chunks):
            predictions = self.find_relevant_chunks(question, top_k=max(ks))
            predicted_chunks = [p['chunk'] for p in predictions]
            
            for k in ks:
                if true_chunk in predicted_chunks[:k]:
                    metrics[f'recall@{k}'] += 1
        
        # Вычисляем средние значения
        for k in ks:
            metrics[f'recall@{k}'] /= total
            
        return metrics