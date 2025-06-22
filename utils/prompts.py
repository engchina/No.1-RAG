"""
Prompt templates for RAG system
"""

# Sub-Query prompt template
SUB_QUERY_PROMPT_TEMPLATE = """
Directly break down the main query into specific, manageable sub-queries. Each sub-query should address a separate aspect of the original query to aid in focused exploration. Avoid including detailed explanations or procedural steps in the sub-queries themselves. Please respond to me in the same language I use for my messages. If I switch languages, please switch your responses accordingly.
"""

# RAG-Fusion prompt template
RAG_FUSION_PROMPT_TEMPLATE = """
Generate a specific number of search queries directly related to the input query, without providing any additional context, introduction, or explanation in the output. Your primary goal is to fulfill the exact request, focusing solely on the content of the queries specified. Please respond to me in the same language I use for my messages. If I switch languages, please switch your responses accordingly.
"""

# HyDE prompt template
HYDE_PROMPT_TEMPLATE = """
Generate hypothetical answers for input queries using the HyDE method, focusing solely on the essence of the queries. Output should be limited to the exact number of requested answers, presented succinctly and without any additional formatting, spacing, or explanatory text. Please respond to me in the same language I use for my messages. If I switch languages, please switch your responses accordingly.
"""

# Step-Back-Prompting template
STEP_BACK_PROMPT_TEMPLATE = """
Generate broader, more abstract questions that step back from the specific details of the input query. These questions should help explore the fundamental concepts and principles underlying the original query. Please respond to me in the same language I use for my messages. If I switch languages, please switch your responses accordingly.
"""

# Unused
"""
1. If no appropriate answer can be found from the context, respond with: "申し訳ありませんが、コンテキストから適切な回答を見つけることができませんでした。別の LLM モデルをお試しいただくか、クエリの内容や設定を少し調整していただくことで解決できるかもしれません。"

- Strict analysis with UTF-8 encoding
"""

# LangGPT RAG prompt template
LANGGPT_RAG_PROMPT_TEMPLATE = """
## Role: Strict Context QA

### Profile
- Author: User
- Version: 0.2
- Language: Multi-language
- Description: A strict context-based question-answering system that uses only the provided context data and responds without any modifications.

### Core Skills
1. Complete context matching search
2. Complete elimination of context modification
3. Standard notification when unable to answer
4. Multi-format output support

## Rules
1. Answers must be 100% dependent on the content within <context></context>
2. Do not perform partial matching or speculation
3. Handle chronological processing when date information is available (prioritize latest information)
4. Maintain strict formatting of citation information
5. You always respond to the user in the Japanese language.

## Workflow
1. Context Analysis Phase
   - Extract metadata (EMBED_ID/SOURCE)
2. Query Matching Phase
   - Apply complete string matching algorithm
   - Prioritize latest date when multiple candidates exist
3. Answer Generation Phase
   - Direct citation of matched data
   - Structured output of citation information
4. Error Handling
   - No match → Standard error message
   - Contradictory data → List factual relationships

## Initialization
As a Strict Context QA system, you must follow the Rules in the specified Language.
The context QA system has been activated. Please provide the following elements:

<context>
{context}
</context>

<query>
{query_text}
</query>
"""

# System message for LLM evaluation
LLM_EVALUATION_SYSTEM_MESSAGE = """
-目標活動-
あなたは「回答評価者」です。

-目標-
あなたの任務は、与えられた回答を標準回答と比較し、その質を評価することです。
以下の各基準について0から10の評点で回答してください：
1.正確さ（0は完全に不正確、10は完全に正確）
2.完全性（0はまったく不完全、10は完全に満足）
3.明確さ（0は非常に不明確、10は非常に明確）
4.簡潔さ（0は非常に冗長、10は最適に簡潔）

評点を付けた後、各評点について簡単な説明を加えてください。
最後に、0から10の総合評価と評価の要約を提供してください。
私のメッセージと同じ言語で返答してください。
もし私が言語を切り替えた場合は、それに応じて返答の言語も切り替えてください。
"""

# Chat system message
CHAT_SYSTEM_MESSAGE = """あなたは役立つアシスタントです。
私のメッセージと同じ言語で返答してください。
もし私が言語を切り替えた場合は、それに応じて返答の言語も切り替えてください。"""

# MarkItDown LLM prompt
MARKITDOWN_LLM_PROMPT = "画像にふさわしい詳細な代替キャプションを書いてください。"

# Query generation prompts
QUERY_GENERATION_PROMPTS = {
    "Sub-Query": {
        "user_template": "Decompose the following query into exactly 3 targeted sub-queries that can be individually explored: {original_query}. Follow the format demonstrated by these few-shot examples: '1. xxx', '2. xxx', '3. xxx'\nOUTPUT:"
    },
    "RAG-Fusion": {
        "user_template": "Generate exactly 3 search queries related to: {original_query}. Follow the format demonstrated by these few-shot examples: '1. xxx', '2. xxx', '3. xxx'\nOUTPUT:"
    },
    "HyDE": {
        "user_template": "Directly generate exactly 3 hypothetical answers for: {original_query}. Follow the format demonstrated by these few-shot examples: '1. xxx', '2. xxx', '3. xxx'\nOUTPUT:"
    },
    "Step-Back-Prompting": {
        "user_template": "Generate exactly 3 broader, more abstract questions that step back from the specific details of: {original_query}. Follow the format demonstrated by these few-shot examples: '1. xxx', '2. xxx', '3. xxx'\nOUTPUT:"
    }
}

def get_sub_query_prompt():
    """Get sub-query prompt template"""
    return SUB_QUERY_PROMPT_TEMPLATE

def get_rag_fusion_prompt():
    """Get RAG-Fusion prompt template"""
    return RAG_FUSION_PROMPT_TEMPLATE

def get_hyde_prompt():
    """Get HyDE prompt template"""
    return HYDE_PROMPT_TEMPLATE

def get_step_back_prompt():
    """Get Step-Back-Prompting template"""
    return STEP_BACK_PROMPT_TEMPLATE

def get_langgpt_rag_prompt(context, query_text, include_citation=False, include_current_time=False, custom_template=None):
    """Get LangGPT RAG prompt with context and query"""
    # Use custom template if provided, otherwise use default template
    template = custom_template if custom_template else LANGGPT_RAG_PROMPT_TEMPLATE

    # Replace double braces with single braces for format() function
    if custom_template:
        template = template.replace('{{context}}', '{context}').replace('{{query_text}}', '{query_text}')

    prompt = template.format(context=context, query_text=query_text)

    # Add citation format if requested
    if include_citation:
        prompt += """
### Citation Format Rules
- Add JSON array immediately after output
- Maintain strict structure (do not use ```json):
[
    {
        "EMBED_ID": <unique identifier>,
        "SOURCE": "<information source>"
    }
]
"""

    # Add time processing if requested
    if include_current_time:
        from datetime import datetime
        current_time = datetime.now().strftime('%Y%m%d')
        prompt += f"""
The current date is {current_time}.
"""

    return prompt.strip()

def get_llm_evaluation_system_message():
    """Get LLM evaluation system message"""
    return LLM_EVALUATION_SYSTEM_MESSAGE

def get_chat_system_message():
    """Get chat system message"""
    return CHAT_SYSTEM_MESSAGE

def get_markitdown_llm_prompt():
    """Get MarkItDown LLM prompt"""
    return MARKITDOWN_LLM_PROMPT

def get_query_generation_prompt(query_type, original_query):
    """Get query generation prompt for specific type"""
    if query_type in QUERY_GENERATION_PROMPTS:
        return QUERY_GENERATION_PROMPTS[query_type]["user_template"].format(original_query=original_query)
    return ""

def update_langgpt_rag_prompt(new_prompt):
    """Update LangGPT RAG prompt template"""
    global LANGGPT_RAG_PROMPT_TEMPLATE
    LANGGPT_RAG_PROMPT_TEMPLATE = new_prompt

def update_llm_evaluation_system_message(new_message):
    """Update LLM evaluation system message"""
    global LLM_EVALUATION_SYSTEM_MESSAGE
    LLM_EVALUATION_SYSTEM_MESSAGE = new_message

# Test function for the custom template functionality
if __name__ == "__main__":
    # Test with default template
    default_result = get_langgpt_rag_prompt("test context", "test query")
    print("Default template test:")
    print(default_result[:100] + "...")

    # Test with custom template
    custom_template = """
Custom RAG Template:
Context: {{context}}
Query: {{query_text}}
Please answer based on the context.
"""
    custom_result = get_langgpt_rag_prompt("test context", "test query", custom_template=custom_template)
    print("\nCustom template test:")
    print(custom_result)
