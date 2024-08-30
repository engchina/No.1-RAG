import re

import spacy
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 加载模型
nlp_ja_ginza = spacy.load('ja_ginza_electra')
nlp_ja_core = spacy.load('ja_core_news_lg')
nlp_en_core = spacy.load('en_core_web_lg')


class QueryText(BaseModel):
    query_text: str
    language: str  # 添加一个语言字段


@app.post("/v1/tokenize")
def split_query(query: QueryText):
    """
    Extracts relevant search texts from the input query text based on the specified language and token tags.
    Filters out common English stopwords.

    Args:
    - query_text (str): The input query text to extract search texts from.
    - language (str): Language indicator ('ja' for Japanese, 'zh' for Chinese, 'en' for English).

    Returns:
    - list: A list of search texts extracted from the input query text, excluding common stopwords.
    """
    print(f"{query.query_text=}")
    final_split_queries = []

    try:
        if query.language == 'ja':
            # 日文分词处理
            # ja_split_queries = [token.text for token in nlp_ja_ginza(query.query_text) if
            #                     any(token.tag_.startswith(tag) for tag in
            #                         ['名詞-数詞', '名詞-普通名詞', '名詞-固有名詞', '動詞-一般'])
            #                     and not re.fullmatch(r'[0-9]+', token.text)]
            ja_split_queries = [token.text for token in nlp_ja_ginza(query.query_text) if
                                any(token.tag_.startswith(tag) for tag in
                                    ['名詞-普通名詞', '名詞-固有名詞'])
                                and not re.fullmatch(r'[0-9]+', token.text)]
            # 英文分词处理
            # en_split_queries = [token.text for token in nlp_en_core(query.query_text) if
            #                     token.pos_ in ['PROPN', 'NOUN', 'VERB', 'NUM'] and re.match(r'^[a-zA-Z]', token.text)]
            en_split_queries = [token.text for token in nlp_en_core(query.query_text) if
                                token.pos_ in ['NOUN'] and re.match(r'^[a-zA-Z]', token.text)]
            final_split_queries = list(set(ja_split_queries + en_split_queries))
        elif query.language == 'en':
            # 只做英文分词处理
            # final_split_queries = [token.text for token in nlp_en_core(query.query_text) if
            #                        token.pos_ in ['PROPN', 'NOUN', 'VERB'] and re.match(r'^[a-zA-Z]', token.text)]
            final_split_queries = [token.text for token in nlp_en_core(query.query_text) if
                                   token.pos_ in ['NOUN'] and re.match(r'^[a-zA-Z]', token.text)]

        # 定义英文停用词列表
        custom_stopwords = {
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'is', 'do', 'this', 'that', 'there', 'and', 'but', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
            'by',
            'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
            'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
            'will',
            'just', 'don', "don't", 'should', "should've", 'now', 'll', 're', 've', 'ain', 'aren',
            "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
            'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
            'shan',
            "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
        }

        # 过滤掉停用词
        final_split_queries = [word for word in set(final_split_queries)
                               if
                               word.lower() not in custom_stopwords and word.lower() not in nlp_ja_ginza.Defaults.stop_words and word.lower() not in nlp_ja_core.Defaults.stop_words and word.lower() not in nlp_en_core.Defaults.stop_words]
        print(f"{final_split_queries=}")
        return final_split_queries

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
