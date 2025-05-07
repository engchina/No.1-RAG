import re

import jieba
import spacy
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 加载模型
nlp_ja_ginza = spacy.load('ja_ginza_electra')
nlp_ja_core = spacy.load('ja_core_news_lg')
nlp_zh_core = spacy.load('zh_core_web_lg')
nlp_en_core = spacy.load('en_core_web_lg')

# 预先合并所有停用词集合
custom_stopwords = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'is', 'do', 'this', 'that', 'there', 'and', 'but', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'll', 're', 've', 'ain', 'aren',
    "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

all_stopwords = set(custom_stopwords)
all_stopwords.update(nlp_ja_ginza.Defaults.stop_words)
all_stopwords.update(nlp_ja_core.Defaults.stop_words)
all_stopwords.update(nlp_en_core.Defaults.stop_words)


class QueryText(BaseModel):
    query_text: str
    language: str  # 添加一个语言字段
    max_tokens: int = 10  # 默认最大返回token数量


def extract_entities(text, nlp_model):
    """提取命名实体"""
    doc = nlp_model(text)
    entities = [ent.text for ent in doc.ents]
    return entities


def limit_results(tokens, max_count=15):
    """限制返回结果数量，优先保留短字符串"""
    if len(tokens) <= max_count:
        return tokens

    # 按字符串长度排序，短的优先保留
    sorted_tokens = sorted(tokens, key=len)
    return sorted_tokens[:max_count]


# 添加一个新的辅助函数来过滤无效token
def is_valid_token(token):
    """检查token是否有效（不含特殊字符组合）"""
    # 过滤掉包含数字和特殊字符组合的token
    if re.search(r'[0-9\[\]\(\)\{\}\<\>]+', token):
        return False
    # 过滤掉过短的token
    if len(token) <= 1:
        return False
    return True


@app.post("/v1/tokenize")
def split_query(query: QueryText):
    """
    Extracts relevant search texts from the input query text based on the specified language and token tags.
    Filters out common stopwords and applies various optimization techniques.

    Args:
    - query_text (str): The input query text to extract search texts from.
    - language (str): Language indicator ('ja' for Japanese, 'zh' for Chinese, 'en' for English).
    - max_tokens (int): Maximum number of tokens to return.

    Returns:
    - list: A list of search texts extracted from the input query text, excluding common stopwords.
    """
    print(f"{query.query_text=}")
    final_split_queries = []
    entities = []

    try:
        if query.language == 'ja':
            # 日文分词处理 - 统一词性筛选为名词类
            ja_split_queries = [token.text for token in nlp_ja_ginza(query.query_text) if
                                any(token.tag_.startswith(tag) for tag in
                                    ['名詞-普通名詞', '名詞-固有名詞'])
                                and not re.search(r'[0-9\[\]\(\)\{\}\<\>]+', token.text)
                                and 1 < len(token.text) < 10]  # 添加词长限制

            # 英文分词处理
            en_split_queries = [token.text for token in nlp_en_core(query.query_text) if
                                token.pos_ in ['NOUN'] and re.match(r'^[a-zA-Z]', token.text)
                                and not re.search(r'[0-9\[\]\(\)\{\}\<\>]+', token.text)
                                and 2 < len(token.text) < 15]  # 添加词长限制

            # 提取实体
            ja_entities = extract_entities(query.query_text, nlp_ja_core)
            entities.extend(ja_entities)

            final_split_queries = list(set(ja_split_queries + en_split_queries))

        elif query.language == 'zh':  # 使用jieba进行中文分词
            jieba_split_queries = list(jieba.cut(query.query_text))

            # 使用spaCy的中文模型处理jieba的分词结果
            docs = nlp_zh_core.pipe(jieba_split_queries)
            zh_split_queries = [token.text for doc in docs for token in doc if
                                token.pos_ in ['NOUN', 'PROPN']  # 只保留名词类
                                and 1 < len(token.text) < 5
                                and not re.search(r'[0-9\[\]\(\)\{\}\<\>]+', token.text)]

            # 英文分词处理（中文文本中可能包含英文）
            en_split_queries = [token.text for token in nlp_en_core(query.query_text) if
                                token.pos_ in ['NOUN'] and re.match(r'^[a-zA-Z]+$', token.text)
                                and not re.search(r'[0-9\[\]\(\)\{\}\<\>]+', token.text)
                                and 2 < len(token.text) < 15]  # 添加词长限制

            # 提取实体
            zh_entities = extract_entities(query.query_text, nlp_zh_core)
            entities.extend(zh_entities)

            final_split_queries = list(set(zh_split_queries + en_split_queries))

        elif query.language == 'en':
            # 只做英文分词处理
            final_split_queries = [token.text for token in nlp_en_core(query.query_text) if
                                   token.pos_ in ['NOUN'] and re.match(r'^[a-zA-Z]', token.text)
                                   and not re.search(r'[0-9\[\]\(\)\{\}\<\>]+', token.text)
                                   and 2 < len(token.text) < 15]  # 添加词长限制

            # 提取实体
            en_entities = extract_entities(query.query_text, nlp_en_core)
            entities.extend(en_entities)

        # 过滤掉停用词 - 使用预先合并的停用词集合
        final_split_queries = [word for word in set(final_split_queries) if word.lower() not in all_stopwords]
        entities = [entity for entity in entities if entity.lower() not in all_stopwords]

        # 添加实体到结果中
        final_split_queries.extend(entities)
        final_split_queries = list(set(final_split_queries))  # 去重
        
        # 额外过滤包含特殊字符的token
        final_split_queries = [token for token in final_split_queries if is_valid_token(token)]

        # 限制返回结果数量
        final_split_queries = limit_results(final_split_queries, query.max_tokens)

        print(f"{final_split_queries=}")
        return final_split_queries

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
