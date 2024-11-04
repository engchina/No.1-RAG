import argparse
import json
import os
import platform
import re
import shutil
import time
from itertools import combinations
from typing import List, Tuple

import cohere
import gradio as gr
import oci
import oracledb
import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv, set_key
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from oracledb import DatabaseError
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from my_langchain_community.vectorstores import MyOracleVS
from utils.common_util import get_dict_value
from utils.generator_util import generate_unique_id

custom_css = """
body {
  font-family: "Noto Sans JP", Arial, sans-serif !important;
}

/* Hide sort buttons at gr.DataFrame */
.sort-button {
    display: none !important;
} 

body gradio-app .tabitem .block{
    background: #fff !important;
}

.gradio-container{
    background: #c4c4c440;
}

.tabitem .form{
border-radius: 3px;
}

.main_Header>span>h1{
    color: #fff;
    text-align: center;
    margin: 0 auto;
    display: block;
}

.tab-nav{
    # border-bottom: none !important;
}

.tab-nav button[role="tab"]{
    color: rgb(96, 96, 96);
    font-weight: 500;
    background: rgb(255, 255, 255);
    padding: 10px 20px;
    border-radius: 4px 4px 0px 0px;
    border: none;
    border-right: 4px solid gray;
    border-radius: 0px;
    min-width: 150px;
}

.tabs .tabitem .tabs .tab-nav button[role="tab"]{
    min-width: 90px;
    padding: 5px;
    border-right: 1px solid #186fb4;
    border-top: 1px solid #186fb4;
    border-bottom: 0.2px solid #fff;
    margin-bottom: -2px;
    z-index: 3;
}


.tabs .tabitem .tabs .tab-nav button[role="tab"]:first-child{
    border-left: 1px solid #186fb4;
        border-top-left-radius: 3px;
}

.tabs .tabitem .tabs .tab-nav button[role="tab"]:last-child{
    border-right: 1px solid #186fb4;
}

.tab-nav button[role="tab"]:first-child{
       border-top-left-radius: 3px;
}

.tab-nav button[role="tab"]:last-child{
        border-top-right-radius: 3px;
    border-right: none;
}
.tabitem{
    background: #fff;
    border-radius: 0px 3px 3px 3px !important;
    box-shadow: rgba(50, 50, 93, 0.25) 0px 2px 5px -1px, rgba(0, 0, 0, 0.3) 0px 1px 3px -1px;
}

.tabitem .tabitem{
    border: 1px solid #196fb4;
    background: #fff;
    border-radius: 0px 3px 3px 3px !important;
}

.tabitem textarea, div.tabitem div.container>.wrap{
    background: #f4f8ffc4;
}

.tabitem .container .wrap {
    border-radius: 3px;
}

.tab-nav button[role="tab"].selected{
    color: #fff;
    background: #196fb4;
    border-bottom: none;
}

.tabitem .inner_tab button[role="tab"]{
   border: 1px solid rgb(25, 111, 180);
   border-bottom: none;
}

.app.gradio-container {
  max-width: 1440px;
}

gradio-app{
    background-image: url("https://objectstorage.ap-tokyo-1.oraclecloud.com/n/sehubjapacprod/b/km_newsletter/o/tmp%2Fmain_bg.png") !important;
    background-size: 100vw 100vh !important;
}

input, textarea{
    border-radius: 3px;
}


.container>input:focus, .container>textarea:focus, .block .wrap .wrap-inner:focus{
    border-radius: 3px;
    box-shadow: rgb(255 246 228 / 63%) 0px 0px 0px 3px, rgb(255 248 236 / 12%) 0px 2px 4px 0px inset !important;
    border-color: rgb(249 169 125 / 87%) !important;
}

.tabitem div>button.primary{
    border: none;
    background: linear-gradient(to bottom right, #ffc679, #f38141);
    color: #fff;
    box-shadow: 2px 2px 2px #0000001f;
    border-radius: 3px;
}

.tabitem div>button.primary:hover{
    border: none;
    background: #f38141;
    color: #fff;
    border-radius: 3px;
    box-shadow: 2px 2px 2px #0000001f;
}


.tabitem div>button.secondary{
    border: none;
    background: linear-gradient(to right bottom, rgb(215 215 217), rgb(194 197 201));
    color: rgb(107 106 106);
    box-shadow: rgba(0, 0, 0, 0.12) 2px 2px 2px;
    border-radius: 3px;
}

.tabitem div>button.secondary:hover{
    border: none;
    background: rgb(175 175 175);
    color: rgb(255 255 255);
    border-radius: 3px;
    box-shadow: rgba(0, 0, 0, 0.12) 2px 2px 2px;
}

.cus_ele1_select .container .wrap:focus-within{
    border-radius: 3px;
    box-shadow: rgb(255 246 228 / 63%) 0px 0px 0px 3px, rgb(255 248 236 / 12%) 0px 2px 4px 0px inset !important;
    border-color: rgb(249 169 125 / 87%) !important;
}

input[type="checkbox"]:checked, input[type="checkbox"]:checked:hover, input[type="checkbox"]:checked:focus {
    border-color: #186fb4;
    background-color: #186fb4;
}

#event_tbl{
    border-radius:3px;
}

#event_tbl .table-wrap{
    border-radius:3px;
}

#event_tbl table thead>tr>th{
    background: #bfd1e0;
        min-width: 90px;
}

#event_tbl table thead>tr>th:first-child{
    border-radius:3px 0px 0px 0px;
}
#event_tbl table thead>tr>th:last-child{
    border-radius:0px 3px 0px 0px;
}


#event_tbl table .cell-wrap span{
    font-size: 0.8rem;
}

#event_tbl table{
    overflow-y: auto;
    overflow-x: auto;
}

#event_exp_tbl .table-wrap{
     border-radius:3px;   
}
#event_exp_tbl table thead>tr>th{
    background: #bfd1e0;
}

.count_t1_text .prose{
    padding: 5px 0px 0px 6px;
}

.count_t1_text .prose>span{
    padding: 0px;
}

.cus_ele1_select .container .wrap:focus-within{
    border-radius: 3px;
    box-shadow: rgb(255 246 228 / 63%) 0px 0px 0px 3px, rgb(255 248 236 / 12%) 0px 2px 4px 0px inset !important;
    border-color: rgb(249 169 125 / 87%) !important;
}

.count_t1_text .prose>span{
    font-size: 0.9rem;
}


footer{
  display: none !important;
}

.sub_Header>span>h3,.sub_Header>span>h2,.sub_Header>span>h4{
    color: #fff;
    font-size: 0.8rem;
    font-weight: normal;
    text-align: center;
    margin: 0 auto;
    padding: 5px;
}

@media (min-width: 1280px) {
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1400px;
    }
}
.gap.svelte-vt1mxs{
    gap: unset;
}

.tabitem .gap.svelte-vt1mxs{
        gap: var(--layout-gap);
}

@media (min-width: 1280px) {
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1400px;
    }
}

"""

# read local .env file
load_dotenv(find_dotenv())

DEFAULT_COLLECTION_NAME = os.environ["DEFAULT_COLLECTION_NAME"]

if platform.system() == 'Linux':
    oracledb.init_oracle_client(lib_dir="/u01/aipoc/instantclient_23_5")

# 初始化一个数据库连接
pool = oracledb.create_pool(
    dsn=os.environ["ORACLE_23AI_CONNECTION_STRING"],
    min=10,
    max=50,
    increment=1
)


def do_auth(username, password):
    dsn = os.environ["ORACLE_23AI_CONNECTION_STRING"]
    pattern = r"^([^/]+)/([^@]+)@"
    match = re.match(pattern, dsn)

    if match:
        if username.lower() == match.group(1).lower() and password == match.group(2):
            return True
    return False


def generate_embedding_response(inputs: List[str]):
    config = oci.config.from_file('~/.oci/config', "DEFAULT")
    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config,
                                                                                             service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                                                                                             retry_strategy=oci.retry.NoneRetryStrategy(),
                                                                                             timeout=(10, 240))
    batch_size = 96
    all_embeddings = []

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]

        embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
        embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=os.environ["OCI_COHERE_EMBED_MODEL"])
        embed_text_detail.inputs = batch
        embed_text_detail.truncate = "NONE"
        embed_text_detail.compartment_id = os.environ["OCI_COMPARTMENT_OCID"]

        embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)

        print(f"Processed batch {i // batch_size + 1} of {(len(inputs) - 1) // batch_size + 1}")

        all_embeddings.extend(embed_text_response.data.embeddings)

    return all_embeddings


def get_doc_list() -> List[Tuple[str, str]]:
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(f"""
SELECT
    json_value(cmetadata, '$.file_name') name,
    id
FROM
    {DEFAULT_COLLECTION_NAME}_collection
ORDER BY name """)
                return [(f"{row[0]}", row[1]) for row in cursor.fetchall()]
            except DatabaseError as de:
                return []


def refresh_doc_list():
    doc_list = get_doc_list()
    return (
        gr.Radio(choices=doc_list, value=""),
        gr.CheckboxGroup(choices=doc_list, value=""),
        gr.CheckboxGroup(choices=doc_list, value="")
    )


def get_server_path(doc_id: str) -> str:
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"""
SELECT json_value(cmetadata, '$.server_path') AS server_path
FROM {DEFAULT_COLLECTION_NAME}_collection
WHERE id = :doc_id """, doc_id=doc_id)
            return cursor.fetchone()[0]


def process_text_chunks(unstructured_chunks):
    chunks = []
    chunk_id = 1
    start_offset = 1

    for chunk in unstructured_chunks:
        chunk_length = len(chunk.text)
        if chunk_length == 0:
            continue
        chunks.append({
            'CHUNK_ID': chunk_id,
            'CHUNK_OFFSET': start_offset,
            'CHUNK_LENGTH': chunk_length,
            'CHUNK_DATA': chunk.text
        })

        # 更新 ID 和偏移量
        chunk_id += 1
        start_offset += chunk_length

    return chunks


async def command_r_task(system_text, query_text, command_r_checkbox):
    if command_r_checkbox:
        command_r_16k = ChatOCIGenAI(
            model_id="cohere.command-r-16k",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "max_tokens": 2048},
        )
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=query_text),
        ]
        start_time = time.time()
        print(f"{start_time=}")
        langfuse_handler = CallbackHandler(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )
        async for chunk in command_r_16k.astream(messages, config={"callbacks": [langfuse_handler],
                                                                   "metadata": {
                                                                       "ls_model_name": "cohere.command-r-16k"}}):
            yield chunk.content
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n推論時間: {inference_time:.2f}秒")
        yield f"\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def command_r_plus_task(system_text, query_text, command_r_plus_checkbox):
    if command_r_plus_checkbox:
        command_r_plus = ChatOCIGenAI(
            model_id="cohere.command-r-plus",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "max_tokens": 2048},
        )
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=query_text),
        ]
        start_time = time.time()
        print(f"{start_time=}")
        langfuse_handler = CallbackHandler(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )
        async for chunk in command_r_plus.astream(messages, config={"callbacks": [langfuse_handler],
                                                                    "metadata": {
                                                                        "ls_model_name": "cohere.command-r-plus"}}):
            yield chunk.content
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n推論時間: {inference_time:.2f}秒")
        yield f"\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def openai_gpt4o_task(system_text, query_text, openai_gpt4o_checkbox):
    if openai_gpt4o_checkbox:
        load_dotenv(find_dotenv())
        openai_gpt4o = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"]
        )
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=query_text),
        ]
        start_time = time.time()
        print(f"{start_time=}")
        langfuse_handler = CallbackHandler(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )
        async for chunk in openai_gpt4o.astream(messages, config={"callbacks": [langfuse_handler]}):
            yield chunk.content
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n推論時間: {inference_time:.2f}秒")
        yield f"\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def openai_gpt4_task(system_text, query_text, openai_gpt4_checkbox):
    if openai_gpt4_checkbox:
        load_dotenv(find_dotenv())
        openai_gpt4 = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"]
        )
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=query_text),
        ]
        start_time = time.time()
        print(f"{start_time=}")
        langfuse_handler = CallbackHandler(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )
        async for chunk in openai_gpt4.astream(messages, config={"callbacks": [langfuse_handler]}):
            yield chunk.content
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n推論時間: {inference_time:.2f}秒")
        yield f"\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def claude_3_opus_task(system_text, query_text, claude_3_opus_checkbox):
    if claude_3_opus_checkbox:
        load_dotenv(find_dotenv())
        claude_3_opus = ChatAnthropic(
            model="claude-3-opus-20240229",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
        )
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=query_text),
        ]
        start_time = time.time()
        print(f"{start_time=}")
        langfuse_handler = CallbackHandler(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )
        async for chunk in claude_3_opus.astream(messages, config={"callbacks": [langfuse_handler]}):
            yield chunk.content
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n推論時間: {inference_time:.2f}秒")
        yield f"\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def claude_3_sonnet_task(system_text, query_text, claude_3_sonnet_checkbox):
    if claude_3_sonnet_checkbox:
        load_dotenv(find_dotenv())
        claude_3_sonnet = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
        )
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=query_text),
        ]
        start_time = time.time()
        print(f"{start_time=}")
        langfuse_handler = CallbackHandler(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )
        async for chunk in claude_3_sonnet.astream(messages, config={"callbacks": [langfuse_handler]}):
            yield chunk.content
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n推論時間: {inference_time:.2f}秒")
        yield f"\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def claude_3_haiku_task(system_text, query_text, claude_3_haiku_checkbox):
    if claude_3_haiku_checkbox:
        load_dotenv(find_dotenv())
        claude_3_haiku = ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
        )
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=query_text),
        ]
        start_time = time.time()
        print(f"{start_time=}")
        langfuse_handler = CallbackHandler(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )
        async for chunk in claude_3_haiku.astream(messages, config={"callbacks": [langfuse_handler]}):
            yield chunk.content
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n推論時間: {inference_time:.2f}秒")
        yield f"\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def chat(system_text, command_r_user_text, command_r_plus_user_text,
               openai_gpt4o_user_text, openai_gpt4_user_text,
               claude_3_opus_user_text, claude_3_sonnet_user_text, claude_3_haiku_user_text,
               command_r_checkbox, command_r_plus_checkbox,
               openai_gpt4o_gen_checkbox, openai_gpt4_gen_checkbox,
               claude_3_opus_checkbox, claude_3_sonnet_checkbox, claude_3_haiku_checkbox):
    command_r_gen = command_r_task(system_text, command_r_user_text, command_r_checkbox)
    command_r_plus_gen = command_r_plus_task(system_text, command_r_plus_user_text, command_r_plus_checkbox)
    openai_gpt4o_gen = openai_gpt4o_task(system_text, openai_gpt4o_user_text, openai_gpt4o_gen_checkbox)
    openai_gpt4_gen = openai_gpt4_task(system_text, openai_gpt4_user_text, openai_gpt4_gen_checkbox)
    claude_3_opus_gen = claude_3_opus_task(system_text, claude_3_opus_user_text, claude_3_opus_checkbox)
    claude_3_sonnet_gen = claude_3_sonnet_task(system_text, claude_3_sonnet_user_text, claude_3_sonnet_checkbox)
    claude_3_haiku_gen = claude_3_haiku_task(system_text, claude_3_haiku_user_text, claude_3_haiku_checkbox)

    responses_status = ["", "", "", "", "", "", ""]
    while True:
        responses = ["", "", "", "", "", "", ""]
        generators = [command_r_gen, command_r_plus_gen, openai_gpt4o_gen, openai_gpt4_gen,
                      claude_3_opus_gen, claude_3_sonnet_gen, claude_3_haiku_gen]

        for i, gen in enumerate(generators):
            try:
                response = await anext(gen)
                if response:
                    if response == "TASK_DONE":
                        responses_status[i] = response
                    else:
                        responses[i] = response
            except StopAsyncIteration:
                pass

        yield tuple(responses)

        if all(response_status == "TASK_DONE" for response_status in responses_status):
            print("All tasks completed with DONE")
            break


def set_chat_llm_answer(llm_answer_checkbox):
    command_r_answer_visible = False
    command_r_plus_answer_visible = False
    openai_gpt4o_answer_visible = False
    openai_gpt4_answer_visible = False
    claude_3_opus_answer_visible = False
    claude_3_sonnet_answer_visible = False
    claude_3_haiku_answer_visible = False
    if "cohere/command-r" in llm_answer_checkbox:
        command_r_answer_visible = True
    if "cohere/command-r-plus" in llm_answer_checkbox:
        command_r_plus_answer_visible = True
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt4o_answer_visible = True
    if "openai/gpt-4" in llm_answer_checkbox:
        openai_gpt4_answer_visible = True
    if "claude/opus" in llm_answer_checkbox:
        claude_3_opus_answer_visible = True
    if "claude/sonnet" in llm_answer_checkbox:
        claude_3_sonnet_answer_visible = True
    if "claude/haiku" in llm_answer_checkbox:
        claude_3_haiku_answer_visible = True
    return (gr.Accordion(visible=command_r_answer_visible), gr.Accordion(visible=command_r_plus_answer_visible),
            gr.Accordion(visible=openai_gpt4o_answer_visible),
            gr.Accordion(visible=openai_gpt4_answer_visible), gr.Accordion(visible=claude_3_opus_answer_visible),
            gr.Accordion(visible=claude_3_sonnet_answer_visible), gr.Accordion(visible=claude_3_haiku_answer_visible))


def set_chat_llm_evaluation(llm_evaluation_checkbox):
    command_r_evaluation_visible = False
    command_r_plus_evaluation_visible = False
    openai_gpt4o_evaluation_visible = False
    openai_gpt4_evaluation_visible = False
    claude_3_opus_evaluation_visible = False
    claude_3_sonnet_evaluation_visible = False
    claude_3_haiku_evaluation_visible = False
    if llm_evaluation_checkbox:
        command_r_evaluation_visible = True
        command_r_plus_evaluation_visible = True
        openai_gpt4o_evaluation_visible = True
        openai_gpt4_evaluation_visible = True
        claude_3_opus_evaluation_visible = True
        claude_3_sonnet_evaluation_visible = True
        claude_3_haiku_evaluation_visible = True
    return (gr.Accordion(visible=command_r_evaluation_visible), gr.Accordion(visible=command_r_plus_evaluation_visible),
            gr.Accordion(visible=openai_gpt4o_evaluation_visible),
            gr.Accordion(visible=openai_gpt4_evaluation_visible),
            gr.Accordion(visible=claude_3_opus_evaluation_visible),
            gr.Accordion(visible=claude_3_sonnet_evaluation_visible),
            gr.Accordion(visible=claude_3_haiku_evaluation_visible))


async def chat_stream(system_text, query_text, llm_answer_checkbox):
    if not llm_answer_checkbox or len(llm_answer_checkbox) == 0:
        raise gr.Error("LLM モデルを選択してください")
    if not query_text:
        raise gr.Error("ユーザー・メッセージを入力してください")
    command_r_user_text = query_text
    command_r_plus_user_text = query_text
    openai_gpt4o_user_text = query_text
    openai_gpt4_user_text = query_text
    claude_3_opus_user_text = query_text
    claude_3_sonnet_user_text = query_text
    claude_3_haiku_user_text = query_text

    command_r_checkbox = False
    command_r_plus_checkbox = False
    openai_gpt4o_checkbox = False
    openai_gpt4_checkbox = False
    claude_3_opus_checkbox = False
    claude_3_sonnet_checkbox = False
    claude_3_haiku_checkbox = False
    if "cohere/command-r" in llm_answer_checkbox:
        command_r_checkbox = True
    if "cohere/command-r-plus" in llm_answer_checkbox:
        command_r_plus_checkbox = True
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt4o_checkbox = True
    if "openai/gpt-4" in llm_answer_checkbox:
        openai_gpt4_checkbox = True
    if "claude/opus" in llm_answer_checkbox:
        claude_3_opus_checkbox = True
    if "claude/sonnet" in llm_answer_checkbox:
        claude_3_sonnet_checkbox = True
    if "claude/haiku" in llm_answer_checkbox:
        claude_3_haiku_checkbox = True
    # ChatOCIGenAI
    command_r_response = ""
    command_r_plus_response = ""
    openai_gpt4o_response = ""
    openai_gpt4_response = ""
    claude_3_opus_response = ""
    claude_3_sonnet_response = ""
    claude_3_haiku_response = ""
    async for r, r_plus, gpt4o, gpt4, opus, sonnet, haiku in chat(system_text, command_r_user_text,
                                                                  command_r_plus_user_text,
                                                                  openai_gpt4o_user_text, openai_gpt4_user_text,
                                                                  claude_3_opus_user_text, claude_3_sonnet_user_text,
                                                                  claude_3_haiku_user_text,
                                                                  command_r_checkbox,
                                                                  command_r_plus_checkbox, openai_gpt4o_checkbox,
                                                                  openai_gpt4_checkbox, claude_3_opus_checkbox,
                                                                  claude_3_sonnet_checkbox, claude_3_haiku_checkbox):
        command_r_response += r
        command_r_plus_response += r_plus
        openai_gpt4o_response += gpt4o
        openai_gpt4_response += gpt4
        claude_3_opus_response += opus
        claude_3_sonnet_response += sonnet
        claude_3_haiku_response += haiku
        yield (command_r_response, command_r_plus_response, openai_gpt4o_response, openai_gpt4_response,
               claude_3_opus_response, claude_3_sonnet_response, claude_3_haiku_response)


def create_oci_cred(user_ocid, tenancy_ocid, fingerprint, private_key_file):
    def process_private_key(private_key_file_path):
        with open(private_key_file_path, 'r') as file:
            lines = file.readlines()

        processed_key = ''.join(line.strip() for line in lines if not line.startswith('--'))
        return processed_key

    if not user_ocid:
        raise gr.Error("User OCIDを入力してください")
    if not tenancy_ocid:
        raise gr.Error("Tenancy OCIDを入力してください")
    if not fingerprint:
        raise gr.Error("Fingerprintを入力してください")
    if not private_key_file:
        raise gr.Error("Private Keyを入力してください")

    user_ocid = user_ocid.strip()
    tenancy_ocid = tenancy_ocid.strip()
    fingerprint = fingerprint.strip()

    # set up OCI config
    if not os.path.exists("/root/.oci"):
        os.makedirs("/root/.oci")
    if not os.path.exists("/root/.oci/config"):
        shutil.copy("./.oci/config", "/root/.oci/config")
    oci_config_path = find_dotenv("/root/.oci/config")
    key_file_path = '/root/.oci/oci_api_key.pem'
    set_key(oci_config_path, "user", user_ocid, quote_mode="never")
    set_key(oci_config_path, "tenancy", tenancy_ocid, quote_mode="never")
    set_key(oci_config_path, "region", 'us-chicago-1', quote_mode="never")
    set_key(oci_config_path, "fingerprint", fingerprint, quote_mode="never")
    set_key(oci_config_path, "key_file", key_file_path, quote_mode="never")
    shutil.copy(private_key_file.name, key_file_path)
    load_dotenv(oci_config_path)

    # set up OCI Credential on database
    private_key = process_private_key(private_key_file.name)

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            try:
                # Define the PL/SQL statement
                append_acl_sql = """
BEGIN
  DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
    host => '*',
    ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                       principal_name => 'admin',
                       principal_type => xs_acl.ptype_db));
END;
                """

                # Execute the PL/SQL statement
                cursor.execute(append_acl_sql)

                drop_oci_cred_sql = "BEGIN dbms_vector.drop_credential('OCI_CRED'); END;"
                cursor.execute(drop_oci_cred_sql)
            except DatabaseError as de:
                print(f"DatabaseError={de}")

            oci_cred = {
                'user_ocid': user_ocid,
                'tenancy_ocid': tenancy_ocid,
                'compartment_ocid': os.environ["OCI_COMPARTMENT_OCID"],
                'private_key': private_key.strip(),
                'fingerprint': fingerprint
            }

            create_oci_cred_sql = """
BEGIN
   dbms_vector.create_credential(
       credential_name => 'OCI_CRED',
       params => json(:json_params)
   );
END; """

            cursor.execute(create_oci_cred_sql, json_params=json.dumps(oci_cred))
            conn.commit()

    create_oci_cred_sql = f"""
-- Append Host ACE    
BEGIN
  DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
    host => '*',
    ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                       principal_name => 'admin',
                       principal_type => xs_acl.ptype_db));
END;

-- Drop Existing OCI Credential
BEGIN dbms_vector.drop_credential('OCI_CRED'); END;

-- Create New OCI Credential
BEGIN
    dbms_vector.create_credential(
        credential_name => 'OCI_CRED',
        params => json('{json.dumps(oci_cred)}')
    );
END; 
"""
    gr.Info("OCI API Keyの設定が完了しました")
    return gr.Accordion(), gr.Textbox(value=create_oci_cred_sql.strip())


def create_cohere_cred(cohere_cred_api_key):
    if not cohere_cred_api_key:
        raise gr.Error("Cohere API Keyを入力してください")
    cohere_cred_api_key = cohere_cred_api_key.strip()
    env_path = find_dotenv()
    os.environ["COHERE_API_KEY"] = cohere_cred_api_key
    set_key(env_path, "COHERE_API_KEY", cohere_cred_api_key, quote_mode="never")
    load_dotenv(env_path)
    gr.Info("Cohere API Keyの設定が完了しました")
    return gr.Textbox(value=cohere_cred_api_key)


def create_openai_cred(openai_cred_base_url, openai_cred_api_key):
    if not openai_cred_base_url:
        raise gr.Error("OpenAI Base URLを入力してください")
    if not openai_cred_api_key:
        raise gr.Error("OpenAI API Keyを入力してください")
    openai_cred_base_url = openai_cred_base_url.strip()
    openai_cred_api_key = openai_cred_api_key.strip()
    env_path = find_dotenv()
    os.environ["OPENAI_BASE_URL"] = openai_cred_base_url
    os.environ["OPENAI_API_KEY"] = openai_cred_api_key
    set_key(env_path, "OPENAI_BASE_URL", openai_cred_base_url, quote_mode="never")
    set_key(env_path, "OPENAI_API_KEY", openai_cred_api_key, quote_mode="never")
    load_dotenv(find_dotenv())
    gr.Info("OpenAI API Keyの設定が完了しました")
    return gr.Textbox(value=openai_cred_base_url), gr.Textbox(value=openai_cred_api_key)


def create_claude_cred(claude_cred_api_key):
    if not claude_cred_api_key:
        raise gr.Error("Claude API Keyを入力してください")
    claude_cred_api_key = claude_cred_api_key.strip()
    env_path = find_dotenv()
    os.environ["ANTHROPIC_API_KEY"] = claude_cred_api_key
    set_key(env_path, "ANTHROPIC_API_KEY", claude_cred_api_key, quote_mode="never")
    load_dotenv(env_path)
    gr.Info("Claude API Keyの設定が完了しました")
    return gr.Textbox(value=claude_cred_api_key)


def create_langfuse_cred(langfuse_cred_secret_key, langfuse_cred_public_key, langfuse_cred_host):
    if not langfuse_cred_secret_key:
        raise gr.Error("Langfuse Secret Keyを入力してください")
    if not langfuse_cred_public_key:
        raise gr.Error("Langfuse Public Keyを入力してください")
    if not langfuse_cred_host:
        raise gr.Error("Langfuse Hostを入力してください")
    langfuse_cred_secret_key = langfuse_cred_secret_key.strip()
    langfuse_cred_public_key = langfuse_cred_public_key.strip()
    langfuse_cred_host = langfuse_cred_host.strip()
    env_path = find_dotenv()
    os.environ["LANGFUSE_SECRET_KEY"] = langfuse_cred_secret_key
    os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_cred_public_key
    os.environ["LANGFUSE_HOST"] = langfuse_cred_host
    set_key(env_path, "LANGFUSE_SECRET_KEY", langfuse_cred_secret_key, quote_mode="never")
    set_key(env_path, "LANGFUSE_PUBLIC_KEY", langfuse_cred_public_key, quote_mode="never")
    set_key(env_path, "LANGFUSE_HOST", langfuse_cred_host, quote_mode="never")
    load_dotenv(env_path)
    gr.Info("LangFuseの設定が完了しました")
    return gr.Textbox(value=langfuse_cred_secret_key), gr.Textbox(value=langfuse_cred_public_key), gr.Textbox(
        value=langfuse_cred_host)


def create_table():
    # Drop the preference if it exists
    check_preference_sql = """
SELECT PRE_NAME FROM CTX_PREFERENCES WHERE PRE_NAME = 'WORLD_LEXER' AND PRE_OWNER = USER
"""

    drop_preference_plsql = """
-- Drop Preference        
BEGIN
  CTX_DDL.DROP_PREFERENCE('world_lexer');
END;
"""

    create_preference_plsql = """
-- Create Preference    
BEGIN
  CTX_DDL.CREATE_PREFERENCE('world_lexer','WORLD_LEXER');
END;
"""

    # Drop the index if it exists
    check_index_sql = f"""
SELECT INDEX_NAME FROM USER_INDEXES WHERE INDEX_NAME = '{DEFAULT_COLLECTION_NAME.upper()}_EMBED_DATA_IDX'
"""

    drop_index_sql = f"""
-- Drop Index
DROP INDEX {DEFAULT_COLLECTION_NAME.upper()}_EMBED_DATA_IDX
"""

    create_index_sql = f"""
-- Create Index
-- CREATE INDEX {DEFAULT_COLLECTION_NAME}_embed_data_idx ON {DEFAULT_COLLECTION_NAME}_embedding(embed_data) INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('LEXER world_lexer sync (on commit)')
CREATE INDEX {DEFAULT_COLLECTION_NAME}_embed_data_idx ON {DEFAULT_COLLECTION_NAME}_embedding(embed_data) INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('LEXER world_lexer sync (every "freq=minutely; interval=1")')
"""

    output_sql_text = f"""
-- Create Collection Table
CREATE TABLE IF NOT EXISTS {DEFAULT_COLLECTION_NAME}_collection (
    id VARCHAR2(200),
    data BLOB,
    cmetadata CLOB
); 
"""

    output_sql_text += f"""
-- Create Embedding Table  
CREATE TABLE IF NOT EXISTS {DEFAULT_COLLECTION_NAME}_embedding (
    doc_id VARCHAR2(200),
    embed_id NUMBER,
    embed_data VARCHAR2(4000),
    embed_vector VECTOR(embedding_dim, FLOAT32),
    cmetadata CLOB
);
"""

    output_sql_text += "\n" + create_preference_plsql.strip() + "\n"
    output_sql_text += "\n" + create_index_sql.strip() + ";"

    # Default config file and profile
    embed = OCIGenAIEmbeddings(
        model_id=os.environ["OCI_COHERE_EMBED_MODEL"],
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=os.environ["OCI_COMPARTMENT_OCID"]
    )

    # Initialize OracleVS
    MyOracleVS(
        client=pool.acquire(),
        embedding_function=embed,
        collection_name=DEFAULT_COLLECTION_NAME,
        distance_strategy=DistanceStrategy.COSINE,
        params={"pre_delete_collection": True}
    )

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.execute(check_preference_sql)
            if cursor.fetchone():
                cursor.execute(drop_preference_plsql)
            else:
                print("Preference 'WORLD_LEXER' does not exist.")
            cursor.execute(create_preference_plsql)

            # cursor.execute(check_index_sql)
            # if cursor.fetchone():
            #     cursor.execute(drop_index_sql)
            # else:
            #     print(f"Index '{DEFAULT_COLLECTION_NAME.upper()}_EMBED_DATA_IDX' does not exist.")
            cursor.execute(create_index_sql)
            conn.commit()

    gr.Info("テーブルの作成が完了しました")
    return gr.Accordion(), gr.Textbox(value=output_sql_text.strip())


def test_oci_cred(test_query_text):
    test_query_vector = ""
    embed_genai_params = {
        "provider": "ocigenai",
        "credential_name": "OCI_CRED",
        "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
        "model": "cohere.embed-multilingual-v3.0"
    }

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            plsql = """
DECLARE
    l_embed_genai_params CLOB := :embed_genai_params;
    l_result SYS_REFCURSOR;
BEGIN
    OPEN l_result FOR
        SELECT et.* 
        FROM dbms_vector_chain.utl_to_embeddings(:text_to_embed, JSON(l_embed_genai_params)) et;
    :result := l_result;
END; """

            result_cursor = cursor.var(oracledb.CURSOR)

            cursor.execute(plsql,
                           embed_genai_params=json.dumps(embed_genai_params),
                           text_to_embed=test_query_text,
                           result=result_cursor)

            # Fetch the results from the ref cursor
            with result_cursor.getvalue() as ref_cursor:
                result_rows = ref_cursor.fetchall()
                for row in result_rows:
                    if isinstance(row, tuple):
                        if isinstance(row[0], oracledb.LOB):
                            lob_content = row[0].read()
                            lob_json = json.loads(lob_content)
                            test_query_vector += str(lob_json["embed_vector"]) + "\n"

    return gr.Textbox(value=test_query_vector)


def load_document(file_path, server_directory):
    if not file_path:
        raise gr.Error("ファイルを選択してください")
    if not os.path.exists(server_directory):
        os.makedirs(server_directory)
    doc_id = generate_unique_id("doc_")
    file_name = os.path.basename(file_path.name)
    file_extension = os.path.splitext(file_name)
    if isinstance(file_extension, tuple):
        file_extension = file_extension[1]
    server_path = os.path.join(server_directory, f"{doc_id}_{file_name}")
    shutil.copy(file_path.name, server_path)

    collection_cmeta = {}
    # if file_extension == ".pdf":
    #     loader = PyMuPDFLoader(file_path.name)
    #     documents = loader.load()
    #     # collection_cmeta = documents[0].metadata
    #     # collection_cmeta.pop('page', None)
    #     contents = "".join(fc.page_content for fc in documents)
    #     pages_count = len(documents)
    # else:
    #     with open(server_path, 'rb') as file:
    #         contents = file.read()
    #     pages_count = 1

    # https://docs.unstructured.io/open-source/core-functionality/overview
    pages_count = 1
    # if file_extension == ".pdf":
    #     elements = partition(filename=server_path,
    #                          strategy='hi_res',
    #                          languages=["jpn", "eng", "chi_sim"]
    #                          )
    # else:
    #     # for issue: https://github.com/Unstructured-IO/unstructured/issues/3396
    #     elements = partition(filename=server_path, strategy='hi_res',
    #                          languages=["jpn", "eng", "chi_sim"],
    #                          skip_infer_table_types=["doc", "docx"])
    elements = partition(filename=server_path, strategy='fast',
                         languages=["jpn", "eng", "chi_sim"],
                         extract_image_block_types=["Table"],
                         extract_image_block_to_payload=False,
                         # skip_infer_table_types=["pdf", "ppt", "pptx", "doc", "docx", "xls", "xlsx"])
                         skip_infer_table_types=["pdf", "jpg", "png", "heic", "doc", "docx"])
    original_contents = "\n\n".join([str(el) for el in elements])
    print(f"{original_contents=}")

    collection_cmeta['file_name'] = file_name
    collection_cmeta['source'] = server_path
    collection_cmeta['server_path'] = server_path

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.setinputsizes(**{"data": oracledb.DB_TYPE_BLOB})
            load_document_sql = f"""
-- (Only for Reference) Insert to table {DEFAULT_COLLECTION_NAME}_collection 
INSERT INTO {DEFAULT_COLLECTION_NAME}_collection(id, data, cmetadata)
VALUES (:id, to_blob(:data), :cmetadata) """
            output_sql_text = load_document_sql.replace(":id", "'" + str(doc_id) + "'")
            output_sql_text = output_sql_text.replace(":data", "'...'")
            output_sql_text = output_sql_text.replace(":cmetadata", "'" + json.dumps(collection_cmeta) + "'") + ";"
            cursor.execute(load_document_sql, {
                'id': doc_id,
                'data': original_contents,
                'cmetadata': json.dumps(collection_cmeta)
            })
            conn.commit()

    #             select_contents_sql = f"""
    # -- Select data from table {DEFAULT_COLLECTION_NAME}_collection
    # SELECT dbms_vector_chain.utl_to_text(dt.data) from {DEFAULT_COLLECTION_NAME}_collection dt WHERE dt.id = :doc_id """
    #             output_sql_text += "\n" + select_contents_sql.replace(":doc_id", "'" + str(doc_id) + "'") + ";"
    #             cursor.execute(select_contents_sql, {'doc_id': doc_id})
    #             result = cursor.fetchone()
    #             contents = result[0].read()

    return gr.Textbox(value=output_sql_text.strip()), gr.Textbox(value=doc_id), gr.Textbox(
        value=str(pages_count)), gr.Textbox(value=original_contents)


def split_document_by_oracle(doc_id, chunks_by, chunks_max_size,
                             chunks_overlap_size,
                             chunks_split_by, chunks_split_by_custom,
                             chunks_language, chunks_normalize,
                             chunks_normalize_options):
    # print(f"{chunks_normalize_options=}")
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            parameter_str = '{"by": "' + chunks_by + '","max": "' + str(
                chunks_max_size) + '", "overlap": "' + str(
                int(int(
                    chunks_max_size) * chunks_overlap_size / 100)) + '", "split": "' + chunks_split_by + '", '
            if chunks_split_by == "CUSTOM":
                parameter_str += '"custom_list": [' + chunks_split_by_custom + '], '
            parameter_str += '"language": "' + chunks_language + '", '
            if chunks_normalize == "NONE" or chunks_normalize == "ALL":
                parameter_str += '"normalize": "' + chunks_normalize + '"}'
            else:
                parameter_str += '"normalize": "' + chunks_normalize + '", "norm_options": [' + ", ".join(
                    ['"' + s + '"' for s in chunks_normalize_options]) + ']}'

            # print(f"{parameter_str}")
            select_chunks_sql = f"""
-- Select chunks            
SELECT
    json_value(ct.column_value, '$.chunk_id')     AS chunk_id,
    json_value(ct.column_value, '$.chunk_offset') AS chunk_offset,
    json_value(ct.column_value, '$.chunk_length') AS chunk_length,
    json_value(ct.column_value, '$.chunk_data')   AS chunk_data
FROM
    {DEFAULT_COLLECTION_NAME}_collection dt, 
    dbms_vector_chain.utl_to_chunks(
        dbms_vector_chain.utl_to_text(dt.data),
        JSON(
            :parameter_str
        )
    ) ct
    CROSS JOIN
        JSON_TABLE ( ct.column_value, '$[*]'
            COLUMNS (
                column_value VARCHAR2 ( 4000 ) PATH '$'
            )
        )
WHERE
    dt.id = :doc_id """
            # cursor.setinputsizes(oracledb.DB_TYPE_VECTOR)
            utl_to_chunks_sql_output = "\n" + select_chunks_sql.replace(':parameter_str',
                                                                        "'" + parameter_str + "'").replace(
                ':doc_id', "'" + str(doc_id) + "'") + ";"
            # print(f"{utl_to_chunks_sql_output=}")
            cursor.execute(select_chunks_sql,
                           [parameter_str, doc_id])
            chunks = []
            for row in cursor:
                chunks.append({'CHUNK_ID': row[0],
                               'CHUNK_OFFSET': row[1], 'CHUNK_LENGTH': row[2], 'CHUNK_DATA': row[3]})

            conn.commit()

    chunks_dataframe = pd.DataFrame(chunks)
    return gr.Textbox(utl_to_chunks_sql_output.strip()), gr.Textbox(value=str(chunks_dataframe.shape[0])), gr.Dataframe(
        value=chunks_dataframe)


def split_document_by_unstructured(doc_id, chunks_by, chunks_max_size,
                                   chunks_overlap_size,
                                   chunks_split_by, chunks_split_by_custom,
                                   chunks_language, chunks_normalize,
                                   chunks_normalize_options):
    if not doc_id:
        raise gr.Error("ドキュメントを選択してください")
    # print(f"{chunks_normalize_options=}")
    output_sql = ""
    server_path = get_server_path(doc_id)

    # Use claude to get table data
    # page_table_documents = parser_pdf(server_path)
    page_table_documents = {}
    # elements = partition(filename=server_path,
    #                      strategy='hi_res',
    #                      languages=["jpn", "eng", "chi_sim"]
    #                      )
    # for issue: https://github.com/Unstructured-IO/unstructured/issues/3396
    elements = partition(filename=server_path, strategy='fast',
                         languages=["jpn", "eng", "chi_sim"],
                         extract_image_block_types=["Table"],
                         extract_image_block_to_payload=False,
                         # skip_infer_table_types=["pdf", "ppt", "pptx", "doc", "docx", "xls", "xlsx"])
                         skip_infer_table_types=["pdf", "jpg", "png", "heic", "doc", "docx"])
    prev_page_number = 0
    table_idx = 1
    for el in elements:
        # print(f"{el.category=}")
        # print(f"{el.text=}")
        # print(f"{el.metadata.page_number=}")
        page_number = el.metadata.page_number
        if prev_page_number != page_number:
            prev_page_number = page_number
            table_idx = 1
        # print(f"{type(el.category)=}")
        if el.category == "Table":
            table_id = f"page_{page_number}_table_{table_idx}"
            print(f"{table_id=}")
            print(f"{page_table_documents=}")
            if page_table_documents:
                page_table_document = get_dict_value(page_table_documents, table_id)
                print(f"{page_table_document=}")
                if page_table_document:
                    page_content = get_dict_value(page_table_document, "page_content")
                    table_to_markdown = get_dict_value(page_table_document, "table_to_markdown")
                    if page_content and table_to_markdown:
                        print(f"Before: {el.text=}")
                        el.text = page_content + "\n" + table_to_markdown + "\n"
                        print(f"After: {el.text=}")
            table_idx += 1

    unstructured_chunks = chunk_by_title(elements, include_orig_elements=True,
                                         max_characters=int(chunks_max_size),
                                         multipage_sections=True, new_after_n_chars=int(chunks_max_size),
                                         overlap=int(float(chunks_max_size) * (float(chunks_overlap_size) / 100)),
                                         overlap_all=True)

    chunks = process_text_chunks(unstructured_chunks)
    chunks_dataframe = pd.DataFrame(chunks)

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            delete_sql = f"""
-- Delete chunks
DELETE FROM {DEFAULT_COLLECTION_NAME}_embedding WHERE doc_id = :doc_id """
            cursor.execute(delete_sql,
                           [doc_id])
            output_sql += delete_sql.replace(':doc_id', "'" + str(doc_id) + "'").lstrip() + ";"

            save_chunks_sql = f"""
-- (Only for Reference) Insert chunks
INSERT INTO {DEFAULT_COLLECTION_NAME}_embedding (
doc_id,
embed_id,
embed_data
)
VALUES (:doc_id, :embed_id, :embed_data) """
            output_sql += "\n" + save_chunks_sql.replace(':doc_id', "'" + str(doc_id) + "'") + ";"
            print(f"{output_sql=}")
            # 准备批量插入的数据
            data_to_insert = [(doc_id, chunk['CHUNK_ID'], chunk['CHUNK_DATA']) for chunk in chunks]

            # 执行批量插入
            cursor.executemany(save_chunks_sql, data_to_insert)
            conn.commit()

    return gr.Textbox(value=output_sql), gr.Textbox(value=str(chunks_dataframe.shape[0])), gr.Dataframe(
        value=chunks_dataframe)


def embed_save_document_by_unstructured(doc_id, chunks_by, chunks_max_size,
                                        chunks_overlap_size,
                                        chunks_split_by, chunks_split_by_custom,
                                        chunks_language, chunks_normalize,
                                        chunks_normalize_options):
    if not doc_id:
        raise gr.Error("ドキュメントを選択してください")
    output_sql = ""
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            select_sql = f"""
SELECT doc_id, embed_id, embed_data FROM {DEFAULT_COLLECTION_NAME}_embedding  WHERE doc_id = :doc_id            
"""
            cursor.execute(select_sql, doc_id=doc_id)
            records = cursor.fetchall()
            embed_datas = [record[2] for record in records]
            embed_vectors = generate_embedding_response(embed_datas)
            update_sql = f"""
UPDATE {DEFAULT_COLLECTION_NAME}_embedding 
SET embed_vector = :embed_vector
WHERE doc_id = :doc_id and embed_id = :embed_id
"""

            # update_sql = f"""
            # -- Update chunks
            # UPDATE {DEFAULT_COLLECTION_NAME}_embedding
            # SET embed_vector = dbms_vector.utl_to_embedding(embed_data, json('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}}'))
            # WHERE doc_id = :doc_id """
            # cursor.execute(update_sql,
            #                [doc_id])
            cursor.setinputsizes(embed_vector=oracledb.DB_TYPE_VECTOR)
            output_sql += update_sql.replace(':doc_id', "'" + str(doc_id) + "'"
                                             ).replace(':embed_id', "'" + str('...') + "'"
                                                       ).replace(':embed_vector', "'" + str('...') + "'").strip() + ";"
            print(f"{output_sql=}")
            cursor.executemany(update_sql,
                               [{'doc_id': record[0], 'embed_id': record[1], 'embed_vector': embed_vector}
                                for record, embed_vector in zip(records, embed_vectors)])
            conn.commit()
    return gr.Textbox(output_sql), gr.Textbox(), gr.Dataframe()


def generate_query(query_text, generate_query_radio):
    generate_query1 = ""
    generate_query2 = ""
    generate_query3 = ""

    if generate_query_radio == "None":
        return gr.Textbox(value=generate_query1), gr.Textbox(value=generate_query2), gr.Textbox(value=generate_query3)

    # chat_llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
    #                       model=os.environ["OPENAI_MODEL_NAME"], temperature=0)
    chat_llm = ChatOCIGenAI(
        model_id="cohere.command-r-plus",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
        model_kwargs={"temperature": 0.0, "max_tokens": 2048},
    )

    # RAG-Fusion
    if generate_query_radio == "Sub-Query":
        # v1
        # sub_query_prompt = ChatPromptTemplate.from_messages([
        #     ("system",
        #      """
        #      You are an advanced assistant that specializes in breaking down complex, multifaceted input queries into more manageable sub-queries.
        #      This approach allows for each aspect of the query to be explored in depth, facilitating a comprehensive and nuanced response.
        #      Your task is to dissect the given query into its constituent elements and generate targeted sub-queries that can each be researched or answered individually,
        #      ensuring that the final response holistically addresses all components of the original query.
        #      """),
        #     ("user",
        #      "Decompose the following query into targeted sub-queries that can be individually explored: {original_query} \n OUTPUT (2 queries): )")
        # ])
        # v2
        sub_query_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
             Directly break down the main query into specific, manageable sub-queries. Each sub-query should address a separate aspect of the original query to aid in focused exploration. Avoid including detailed explanations or procedural steps in the sub-queries themselves. Please respond to me in the same language I use for my messages. If I switch languages, please switch your responses accordingly.
             """),
            ("user",
             "Divide the query into exactly 3 distinct sub-queries for focused analysis: {original_query}. Follow the format demonstrated by these few-shot examples: '1. xxx', '2. xxx', '3. xxx' \nOUTPUT:")
        ])

        generate_sub_queries_chain = (
            sub_query_prompt | chat_llm | StrOutputParser() | (lambda x: x.split("\n"))
        )
        sub_queries = generate_sub_queries_chain.invoke({"original_query": query_text})
        print(f"{sub_queries=}")

        if isinstance(sub_queries, list):
            generate_query1 = re.sub(r'^1\. ', '', sub_queries[0])
            generate_query2 = re.sub(r'^2\. ', '', sub_queries[1])
            generate_query3 = re.sub(r'^3\. ', '', sub_queries[2])
    elif generate_query_radio == "RAG-Fusion":
        # v1
        # rag_fusion_prompt = ChatPromptTemplate.from_messages([
        #     ("system",
        #      "You are a helpful assistant that generates multiple similary search queries based on a single input query."),
        #     ("user", "Generate multiple search queries related to: {original_query} \n OUTPUT (2 queries):")
        # ])
        # v2
        rag_fusion_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
             Generate a specific number of search queries directly related to the input query, without providing any additional context, introduction, or explanation in the output. Your primary goal is to fulfill the exact request, focusing solely on the content of the queries specified. Please respond to me in the same language I use for my messages. If I switch languages, please switch your responses accordingly.
             """),
            ("user",
             "Generate exactly 3 search queries related to: {original_query}. Follow the format demonstrated by these few-shot examples: '1. xxx', '2. xxx', '3. xxx'\nOUTPUT:")
        ])

        generate_rag_fusion_queries_chain = (
            rag_fusion_prompt | chat_llm | StrOutputParser() | (lambda x: x.split("\n"))
        )
        rag_fusion_queries = generate_rag_fusion_queries_chain.invoke({"original_query": query_text})
        print(f"{rag_fusion_queries=}")

        if isinstance(rag_fusion_queries, list):
            generate_query1 = re.sub(r'^1\. ', '', rag_fusion_queries[0])
            generate_query2 = re.sub(r'^2\. ', '', rag_fusion_queries[1])
            generate_query3 = re.sub(r'^3\. ', '', rag_fusion_queries[2])
    elif generate_query_radio == "HyDE":
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
             Generate hypothetical answers for input queries using the HyDE method, focusing solely on the essence of the queries. Output should be limited to the exact number of requested answers, presented succinctly and without any additional formatting, spacing, or explanatory text. Please respond to me in the same language I use for my messages. If I switch languages, please switch your responses accordingly.
             """),
            ("user",
             "Directly generate exactly 3 hypothetical answers for: {original_query}. Follow the format demonstrated by these few-shot examples: '1. xxx', '2. xxx', '3. xxx'\nOUTPUT:")
        ])

        generate_hyde_answers_chain = (
            hyde_prompt | chat_llm | StrOutputParser() | (lambda x: x.split("\n"))
        )
        hyde_answers = generate_hyde_answers_chain.invoke({"original_query": query_text})
        print(f"{hyde_answers=}")

        if isinstance(hyde_answers, list):
            generate_query1 = re.sub(r'^1\. ', '', hyde_answers[0])
            generate_query2 = re.sub(r'^2\. ', '', hyde_answers[1])
            generate_query3 = re.sub(r'^3\. ', '', hyde_answers[2])
    elif generate_query_radio == "Step-Back-Prompting":
        step_back_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
             Apply the Step Back Prompt technique by generating three broader, abstracted questions from the original query. These should open up wider avenues for exploration and inference, tapping into underlying principles or broader concepts related to the query. Ensure these step back questions are directly linked to the essence of the original query, providing a foundation for a deeper understanding. Please respond to me in the same language I use for my messages. If I switch languages, please switch your responses accordingly.
             """),
            ("user",
             "Generate exactly 3 step back questions based on the original topic: {original_query}. Format as direct statements without introductory phrases. Follow the format demonstrated by these few-shot examples: '1. xxx', '2. xxx', '3. xxx'\nOUTPUT:")
        ])

        generate_step_back_queries_chain = (
            step_back_prompt | chat_llm | StrOutputParser() | (lambda x: x.split("\n"))
        )
        step_back_queries = generate_step_back_queries_chain.invoke({"original_query": query_text})
        print(f"{step_back_queries=}")

        if isinstance(step_back_queries, list):
            generate_query1 = re.sub(r'^1\. ', '', step_back_queries[0])
            generate_query2 = re.sub(r'^2\. ', '', step_back_queries[1])
            generate_query3 = re.sub(r'^3\. ', '', step_back_queries[2])
    elif generate_query_radio == "Customized-Multi-Step-Query":
        select_multi_step_query_sql = f"""
                SELECT json_value(dt.cmetadata, '$.file_name') name, dc.embed_id embed_id, dc.embed_data embed_data, dc.doc_id doc_id
                FROM {DEFAULT_COLLECTION_NAME}_embedding dc, {DEFAULT_COLLECTION_NAME}_collection dt
                WHERE dc.doc_id = dt.id
                ORDER BY vector_distance(dc.embed_vector , (
                        SELECT to_vector(et.embed_vector) embed_vector
                        FROM
                            dbms_vector_chain.utl_to_embeddings(:query_text, JSON('{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}')) t,
                            JSON_TABLE ( t.column_value, '$[*]'
                                    COLUMNS (
                                        embed_id NUMBER PATH '$.embed_id',
                                        embed_data VARCHAR2 ( 4000 ) PATH '$.embed_data',
                                        embed_vector CLOB PATH '$.embed_vector'
                                    )
                                )
                            et), COSINE)
            """
        select_multi_step_query_sql += "FETCH FIRST 3 ROWS ONLY"
        # Prepare parameters for SQL execution
        multi_step_query_params = {
            "query_text": query_text
        }

        # For debugging: Print the final SQL command.
        # Assuming complete_sql is your SQL string with placeholders like :extend_around_chunk_size, :doc_ids, etc.
        query_sql_output = select_multi_step_query_sql

        # Manually replace placeholders with parameter values for debugging
        for key, value in multi_step_query_params.items():
            placeholder = f":{key}"
            # For the purpose of display, ensure the value is properly quoted if it's a string
            display_value = f"'{value}'" if isinstance(value, str) else str(value)
            query_sql_output = query_sql_output.replace(placeholder, display_value)

        # Now query_sql_output contains the SQL command with parameter values inserted
        print(f"\nQUERY_SQL_OUTPUT:\n{query_sql_output}")
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_multi_step_query_sql, multi_step_query_params)
                multi_step_queries = []
                for row in cursor:
                    # print(f"row: {row}")
                    multi_step_queries.append(row[2])
                print(f"{multi_step_queries=}")

        if isinstance(multi_step_queries, list):
            generate_query1 = multi_step_queries[0]
            generate_query2 = multi_step_queries[1]
            generate_query3 = multi_step_queries[2]

    return gr.Textbox(value=generate_query1), gr.Textbox(value=generate_query2), gr.Textbox(value=generate_query3)


def search_document(reranker_model_radio_input,
                    reranker_top_k_slider_input,
                    reranker_threshold_slider_input,
                    threshold_value_slider_input,
                    top_k_slider_input, doc_id_all_checkbox_input,
                    doc_id_checkbox_group_input, text_search_checkbox_input, text_search_k_slider_input,
                    query_text_input, sub_query1_text_input, sub_query2_text_input,
                    sub_query3_text_input, partition_by_k_slider_input, answer_by_one_checkbox_input,
                    extend_first_chunk_size_input,
                    extend_around_chunk_size_input):
    """
    Retrieve relevant splits for any question using similarity search.
    This is simply "top K" retrieval where we select documents based on embedding similarity to the query.
    """

    def cut_lists(lists, limit=10):
        if not lists or len(lists) == 0:
            return lists
        if len(lists) > limit:
            # lists = random.sample(lists, limit)
            if limit == 1:
                return lists[0]
            half_limit = limit // 2
            if limit % 2 == 0:
                lists = lists[:half_limit] + lists[-half_limit:]
            else:
                lists = lists[:half_limit + 1] + lists[-half_limit:]
        return lists

    def generate_combinations(words_list):
        sampled_list = []
        n = len(words_list)
        threshold = 0.75
        if n < 3:
            sampled_list = words_list
            return sampled_list
        min_required = int(n * threshold)
        print(f"{n=}")
        print(f"{min_required=}")
        sampled_list += list(combinations(words_list, min_required))
        return sampled_list

    def process_lists(lists):
        contains_sql_list = []
        for lst in lists:
            if isinstance(lst, str):
                contains_sql_list.append(f"contains(dc.embed_data, '{lst}') > 0")
            else:
                temp = " and ".join(str(item) for item in lst)
                if temp and temp not in contains_sql_list:
                    contains_sql_list.append(f"contains(dc.embed_data, '{temp}') > 0")
        return ' OR '.join(contains_sql_list)

    def format_keywords(x):
        if len(x) > 0:
            formatted = '[' + ', '.join(x) + ']'
            return f"({len(x)}){formatted}"
        else:
            return "(0)[No Match]"

    def replace_newlines(text):
        if '\n\n' in text:
            text = text.replace('\n\n', '\n')
            return replace_newlines(text)
        else:
            return text

    query_text_input = query_text_input.strip()
    sub_query1_text_input = sub_query1_text_input.strip()
    sub_query2_text_input = sub_query2_text_input.strip()
    sub_query3_text_input = sub_query3_text_input.strip()
    command_r_result = "Not queried"
    command_r_plus_result = "Not queried."
    gpt4o_result = "Not queried."
    gpt4_result = "Not queried."
    opus_result = "Not queried."
    sonnet_result = "Not queried."
    haiku_result = "Not queried."

    # Use OracleAIVector
    unranked_docs = []
    threshold_value = threshold_value_slider_input
    top_k = top_k_slider_input
    doc_ids_str = "'" + "','".join([str(doc_id) for doc_id in doc_id_checkbox_group_input if doc_id]) + "'"
    print(f"{doc_ids_str=}")
    with_sql = """
    -- Select data
    WITH offsets AS (
            SELECT level - (:extend_around_chunk_size / 2 + 1) AS offset
            FROM dual
            CONNECT BY level <= (:extend_around_chunk_size + 1)
    ),
    selected_embed_ids AS 
    ( 
    """
    where_sql = """
                    WHERE 1 = 1 """
    where_threshold_sql = """
                    AND vector_distance(dc.embed_vector, (
                            SELECT to_vector(et.embed_vector) embed_vector
                            FROM
                                dbms_vector_chain.utl_to_embeddings(
                                        :query_text, 
                                        JSON('{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}')) t,
                                JSON_TABLE ( t.column_value, '$[*]'
                                        COLUMNS (
                                            embed_id NUMBER PATH '$.embed_id',
                                            embed_data VARCHAR2 ( 4000 ) PATH '$.embed_data',
                                            embed_vector CLOB PATH '$.embed_vector'
                                        )
                                    )
                                et), COSINE
                            ) <= :threshold_value """
    if not doc_id_all_checkbox_input:
        # where_sql += """
        #         AND dc.doc_id IN (
        #             SELECT CAST(REGEXP_SUBSTR(:doc_ids, '([^,]+)', 1, LEVEL) AS NUMBER)
        #             FROM DUAL
        #             CONNECT BY LEVEL <= LENGTH(:doc_ids) - LENGTH(REPLACE(:doc_ids, ',', '')) + 1
        #         ) """
        where_sql += """
                    AND dc.doc_id IN (
                        SELECT TRIM(BOTH '''' FROM REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL)) AS doc_id
                        FROM DUAL
                        CONNECT BY REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL) IS NOT NULL
                    ) """
    # v4
    base_sql = f"""
                    SELECT dc.doc_id doc_id, dc.embed_id embed_id, vector_distance(dc.embed_vector, (
                            SELECT to_vector(et.embed_vector) embed_vector
                            FROM
                                dbms_vector_chain.utl_to_embeddings(
                                        :query_text, 
                                        JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}}')) t,
                                JSON_TABLE ( t.column_value, '$[*]'
                                        COLUMNS (
                                            embed_id NUMBER PATH '$.embed_id',
                                            embed_data VARCHAR2 ( 4000 ) PATH '$.embed_data',
                                            embed_vector CLOB PATH '$.embed_vector'
                                        )
                                    )
                                et), COSINE
                            ) vector_distance
                    FROM {DEFAULT_COLLECTION_NAME}_embedding dc """ + where_sql + where_threshold_sql + """
                    ORDER BY vector_distance """
    base_sql += """
                    FETCH FIRST :partition_by PARTITIONS BY doc_id, :top_k ROWS ONLY 
    """ if partition_by_k_slider_input > 0 else """
                    FETCH FIRST :top_k ROWS ONLY 
    """
    select_sql = f"""
    ),
    selected_embed_id_doc_ids AS 
    (
            SELECT DISTINCT s.embed_id + o.offset adjusted_embed_id, s.doc_id doc_id
            FROM selected_embed_ids s
            CROSS JOIN offsets o
    ),
    selected_results AS 
    (
            SELECT s1.adjusted_embed_id adjusted_embed_id, s1.doc_id doc_id, NVL(s2.vector_distance, 999999.0) vector_distance
            FROM selected_embed_id_doc_ids s1
            LEFT JOIN selected_embed_ids s2
            ON s1.adjusted_embed_id = s2.embed_id AND s1.doc_id = s2.doc_id
    ),
    aggregated_results AS
    (
            SELECT json_value(dt.cmetadata, '$.file_name') name, dc.embed_id embed_id, dc.embed_data embed_data, dc.doc_id doc_id, MIN(s.vector_distance) vector_distance
            FROM selected_results s, {DEFAULT_COLLECTION_NAME}_embedding dc, {DEFAULT_COLLECTION_NAME}_collection dt
            WHERE s.adjusted_embed_id = dc.embed_id AND s.doc_id = dt.id and dc.doc_id = dt.id  
            GROUP BY dc.doc_id, name, dc.embed_id, dc.embed_data
            ORDER BY vector_distance, dc.doc_id, dc.embed_id
    ),
    ranked_data AS (
        SELECT
            ar.embed_data,
            ar.doc_id,
            ar.embed_id,
            ar.vector_distance,
            ar.name,
            LAG(ar.embed_id) OVER (PARTITION BY ar.doc_id ORDER BY ar.embed_id) AS prev_embed_id,
            LEAD(ar.embed_id) OVER (PARTITION BY ar.doc_id ORDER BY ar.embed_id) AS next_embed_id
        FROM
            aggregated_results ar
    ),
    grouped_data AS (
        SELECT
            rd.embed_data,
            rd.doc_id,
            rd.embed_id,
            rd.vector_distance,
            rd.name,
            CASE 
                WHEN rd.prev_embed_id IS NULL OR rd.embed_id - rd.prev_embed_id > 1 THEN 1
                ELSE 0
            END AS new_group
        FROM
            ranked_data rd
    ),
    groups_marked AS (
        SELECT
            gd.embed_data,
            gd.doc_id,
            gd.embed_id,
            gd.vector_distance,
            gd.name,
            SUM(gd.new_group) OVER (PARTITION BY gd.doc_id ORDER BY gd.embed_id) AS group_id
        FROM
            grouped_data gd
    ),
    aggregated_data AS (
        SELECT
            RTRIM(XMLCAST(XMLAGG(XMLELEMENT(e, ad.embed_data || ',') ORDER BY ad.embed_id) AS CLOB), ',') AS combined_embed_data,
            ad.doc_id,
            MIN(ad.embed_id) AS min_embed_id,
            MIN(ad.vector_distance) AS min_vector_distance,
            ad.name
        FROM
            groups_marked ad
        GROUP BY
            ad.doc_id, ad.group_id, ad.name
    )
    SELECT
        ad.name,
        ad.min_embed_id AS embed_id,
        ad.combined_embed_data,
        ad.doc_id,
        ad.min_vector_distance AS vector_distance
    FROM
        aggregated_data ad
    ORDER BY
        ad.min_vector_distance
    """
    #         ORDER BY dc.doc_id, dc.embed_id

    query_texts = [":query_text"]
    if sub_query1_text_input:
        query_texts.append(":sub_query1")
    if sub_query2_text_input:
        query_texts.append(":sub_query2")
    if sub_query3_text_input:
        query_texts.append(":sub_query3")
    print(f"{query_texts=}")

    search_texts = []
    if text_search_checkbox_input:
        # Generate the combined SQL based on available query texts
        search_texts = requests.post(os.environ["GINZA_API_ENDPOINT"],
                                     json={'query_text': query_text_input, 'language': 'ja'}).json()
        search_text = ""
        if search_texts and len(search_texts) > 0:
            search_texts = cut_lists(search_texts, text_search_k_slider_input)
            generated_combinations = generate_combinations(search_texts)
            search_text = process_lists(generated_combinations)
            # search_texts = cut_lists(search_texts, text_search_k_slider_input)
            # search_text = process_lists(search_texts)
        if len(search_text) > 0:
            where_sql += """
                        AND (""" + search_text + """) """
            full_text_search_sql = f"""
                        SELECT dc.doc_id doc_id, dc.embed_id embed_id, vector_distance(dc.embed_vector, (
                                SELECT to_vector(et.embed_vector) embed_vector
                                FROM
                                    dbms_vector_chain.utl_to_embeddings(
                                            :query_text, 
                                            JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}}')) t,
                                    JSON_TABLE ( t.column_value, '$[*]'
                                            COLUMNS (
                                                embed_id NUMBER PATH '$.embed_id',
                                                embed_data VARCHAR2 ( 4000 ) PATH '$.embed_data',
                                                embed_vector CLOB PATH '$.embed_vector'
                                            )
                                        )
                                    et), COSINE
                                ) vector_distance
                        FROM {DEFAULT_COLLECTION_NAME}_embedding dc """ + where_sql
            complete_sql = (with_sql + """ 
                UNION 
        """.join(
                f"        ({base_sql.replace(':query_text', one_query_text)}        )" for one_query_text in
                query_texts) + """ 
                UNION 
                ( """
                            + full_text_search_sql + """
                ) """
                            + select_sql)
    else:
        print(f"{query_texts=}")
        complete_sql = with_sql + """ 
            UNION 
    """.join(
            f"        ({base_sql.replace(':query_text', one_query_text)}        )" for one_query_text in
            query_texts) + select_sql
        print(f"{complete_sql=}")

    # Prepare parameters for SQL execution
    params = {
        "extend_around_chunk_size": extend_around_chunk_size_input,
        "query_text": query_text_input,
        "threshold_value": threshold_value,
        "top_k": top_k
    }
    if not doc_id_all_checkbox_input:
        params["doc_ids"] = doc_ids_str
    if partition_by_k_slider_input > 0:
        params["partition_by"] = partition_by_k_slider_input

    # Add sub-query texts if they exist
    if sub_query1_text_input:
        params["sub_query1"] = sub_query1_text_input
    if sub_query2_text_input:
        params["sub_query2"] = sub_query2_text_input
    if sub_query3_text_input:
        params["sub_query3"] = sub_query3_text_input

    # For debugging: Print the final SQL command.
    # Assuming complete_sql is your SQL string with placeholders like :extend_around_chunk_size, :doc_ids, etc.
    query_sql_output = complete_sql

    # Manually replace placeholders with parameter values for debugging
    for key, value in params.items():
        if key == "doc_ids":
            value = "''" + "'',''".join([str(doc_id) for doc_id in doc_id_checkbox_group_input if doc_id]) + "''"
        placeholder = f":{key}"
        # For the purpose of display, ensure the value is properly quoted if it's a string
        display_value = f"'{value}'" if isinstance(value, str) else str(value)
        query_sql_output = query_sql_output.replace(placeholder, display_value)
    query_sql_output = query_sql_output.strip() + ";"

    # Now query_sql_output contains the SQL command with parameter values inserted
    print(f"\nQUERY_SQL_OUTPUT:\n{query_sql_output}")

    with (pool.acquire() as conn):
        with conn.cursor() as cursor:
            cursor.execute(complete_sql, params)
            for row in cursor:
                print(f"row: {row}")
                unranked_docs.append([row[0], row[1], row[2].read(), row[3], row[4]])

            if len(unranked_docs) == 0:
                docs_dataframe = pd.DataFrame(
                    columns=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"])

                return (gr.Textbox(value=query_sql_output.strip()), gr.Markdown("**検索結果数**: " + str(
                    docs_dataframe.shape[0]) + "   |   **検索キーワード**: (" + str(
                    len(search_texts)) + ")[" + ', '.join(
                    search_texts) + "]", visible=True),
                        gr.Dataframe(value=docs_dataframe, wrap=True,
                                     headers=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"],
                                     column_widths=["4%", "68%", "6%", "8%", "6%", "8%"]),
                        gr.Textbox(command_r_result), gr.Textbox(command_r_plus_result), gr.Textbox(gpt4o_result),
                        gr.Textbox(gpt4_result), gr.Textbox(opus_result), gr.Textbox(sonnet_result),
                        gr.Textbox(haiku_result), gr.Textbox(sub_query1_text_input), gr.Textbox(sub_query2_text_input),
                        gr.Textbox(sub_query3_text_input))

            # ToDo: In case of error
            if len(unranked_docs) > 999:
                unranked_docs = unranked_docs[:1000]
            if 'cohere/rerank' in reranker_model_radio_input:
                unranked = []
                for doc in unranked_docs:
                    unranked.append(doc[2])
                cohere_reranker = reranker_model_radio_input.split('/')[1]
                print(f"{os.environ['COHERE_API_KEY']=}")
                cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
                ranked_results = cohere_client.rerank(query=query_text_input,
                                                      documents=unranked,
                                                      top_n=len(unranked),
                                                      model=cohere_reranker)
                ranked_scores = [0.0] * len(unranked_docs)
                for result in ranked_results.results:
                    ranked_scores[result.index] = result.relevance_score
                docs_data = [{'CONTENT': doc[2],
                              'EMBED_ID': doc[1],
                              'SOURCE': str(doc[3]) + ":" + doc[0],
                              'DISTANCE': '-' if str(doc[4]) == '999999.0' else str(doc[4]),
                              'SCORE': ce_score} for doc, ce_score in zip(unranked_docs, ranked_scores)]
                docs_dataframe = pd.DataFrame(docs_data)
                docs_dataframe = docs_dataframe[docs_dataframe['SCORE'] >= float(reranker_threshold_slider_input)
                                                ].sort_values(by='SCORE', ascending=False).head(
                    reranker_top_k_slider_input)
            else:
                docs_data = [{'CONTENT': doc[2],
                              'EMBED_ID': doc[1],
                              'SOURCE': str(doc[3]) + ":" + doc[0],
                              'DISTANCE': '-' if str(doc[4]) == '999999.0' else str(doc[4]),
                              'SCORE': '-'} for doc in unranked_docs]
                docs_dataframe = pd.DataFrame(docs_data)

            if extend_first_chunk_size_input > 0:
                filtered_doc_ids = "'" + "','".join(
                    [source.split(':')[0] for source in docs_dataframe['SOURCE'].tolist()]) + "'"
                print(f"{filtered_doc_ids=}")
                select_extend_first_chunk_sql = f"""
SELECT 
        json_value(dt.cmetadata, '$.file_name') name,
        MIN(dc.embed_id) embed_id,
        RTRIM(XMLCAST(XMLAGG(XMLELEMENT(e, dc.embed_data || ',') ORDER BY dc.embed_id) AS CLOB), ',') AS embed_data,
        dc.doc_id doc_id,
        '999999.0' vector_distance
FROM 
        {DEFAULT_COLLECTION_NAME}_embedding dc, {DEFAULT_COLLECTION_NAME}_collection dt
WHERE 
        dc.doc_id = dt.id AND 
        dc.doc_id IN (:filtered_doc_ids) AND 
        dc.embed_id <= :extend_first_chunk_size
GROUP BY
        dc.doc_id, name         
ORDER 
        BY dc.doc_id
            """
                select_extend_first_chunk_sql = (select_extend_first_chunk_sql
                                                 .replace(':filtered_doc_ids', filtered_doc_ids)
                                                 .replace(':extend_first_chunk_size',
                                                          str(extend_first_chunk_size_input)))
                print(f"{select_extend_first_chunk_sql=}")
                query_sql_output += "\n" + select_extend_first_chunk_sql.strip()
                cursor.execute(select_extend_first_chunk_sql)
                first_chunks_df = pd.DataFrame(columns=docs_dataframe.columns)
                for row in cursor:
                    new_data = pd.DataFrame(
                        {'CONTENT': row[2].read(), 'EMBED_ID': row[1], 'SOURCE': str(row[3]) + ":" + row[0],
                         'DISTANCE': '-', 'SCORE': '-'},
                        index=[2])
                    first_chunks_df = pd.concat([new_data, first_chunks_df], ignore_index=True)
                print(f"{first_chunks_df=}")

                # 创建一个空的DataFrame,用于存储更新后的数据
                updated_df = pd.DataFrame(columns=docs_dataframe.columns)

                # 记录每个SOURCE的初始插入位置
                insert_positions = {}

                # 遍历原始数据的每一行
                for index, row in docs_dataframe.iterrows():
                    source = row['SOURCE']

                    # 如果当前SOURCE还没有记录初始插入位置,则将其初始化为当前位置
                    if source not in insert_positions:
                        insert_positions[source] = len(updated_df)

                    # 找到新数据中与当前SOURCE相同的行
                    same_source_new_data = first_chunks_df[first_chunks_df['SOURCE'] == source]

                    # 遍历新数据中与当前SOURCE相同的行
                    for _, new_row in same_source_new_data.iterrows():
                        # 在当前行之前插入新数据
                        updated_df = pd.concat([updated_df[:insert_positions[source]],
                                                pd.DataFrame(new_row).T,
                                                updated_df[insert_positions[source]:]])

                        # 更新当前SOURCE的插入位置
                        insert_positions[source] += 1

                    # 将当前行添加到updated_df中
                    updated_df = pd.concat([updated_df[:insert_positions[source]],
                                            pd.DataFrame(row).T,
                                            updated_df[insert_positions[source]:]])

                    # 更新当前SOURCE的插入位置
                    insert_positions[source] += 1
                docs_dataframe = updated_df

            conn.commit()

            docs_dataframe['KEY_WORDS'] = docs_dataframe['CONTENT'].apply(
                lambda x: [word for word in search_texts if word.lower() in x.lower()])
            docs_dataframe['KEY_WORDS'] = docs_dataframe['KEY_WORDS'].apply(format_keywords)
            docs_dataframe['CONTENT'] = docs_dataframe['CONTENT'].apply(replace_newlines)

            if answer_by_one_checkbox_input:
                docs_dataframe = docs_dataframe[docs_dataframe['SOURCE'] == docs_dataframe['SOURCE'].iloc[0]]

            docs_dataframe = docs_dataframe.reset_index(drop=True)
            docs_dataframe.insert(0, 'NO', pd.Series(range(1, len(docs_dataframe) + 1)))
            print(f"{docs_dataframe=}")

            if len(docs_dataframe) == 0:
                docs_dataframe = pd.DataFrame(
                    columns=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"])

            return (gr.Textbox(value=query_sql_output.strip()), gr.Markdown("**検索結果数**: " + str(
                docs_dataframe.shape[0]) + "   |   **検索キーワード**: (" + str(len(search_texts)) + ")[" + ', '.join(
                search_texts) + "]", visible=True),
                    gr.Dataframe(value=docs_dataframe, wrap=True,
                                 headers=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"],
                                 column_widths=["4%", "62%", "6%", "8%", "6%", "6%", "8%"]))


async def chat_document(search_result,
                        llm_answer_checkbox,
                        query_text):
    query_text = query_text.strip()

    command_r_response = ""
    command_r_plus_response = ""
    openai_gpt4o_response = ""
    openai_gpt4_response = ""
    claude_3_opus_response = ""
    claude_3_sonnet_response = ""
    claude_3_haiku_response = ""

    command_r_checkbox = False
    command_r_plus_checkbox = False
    openai_gpt4o_checkbox = False
    openai_gpt4_checkbox = False
    claude_3_opus_checkbox = False
    claude_3_sonnet_checkbox = False
    claude_3_haiku_checkbox = False
    if "cohere/command-r" in llm_answer_checkbox:
        command_r_checkbox = True
    if "cohere/command-r-plus" in llm_answer_checkbox:
        command_r_plus_checkbox = True
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt4o_checkbox = True
    if "openai/gpt-4" in llm_answer_checkbox:
        openai_gpt4_checkbox = True
    if "claude/opus" in llm_answer_checkbox:
        claude_3_opus_checkbox = True
    if "claude/sonnet" in llm_answer_checkbox:
        claude_3_sonnet_checkbox = True
    if "claude/haiku" in llm_answer_checkbox:
        claude_3_haiku_checkbox = True

    context = '\n'.join(search_result['CONTENT'].astype(str).values)
    print(f"{context=}")
    # system_text = f"""
    #         Use the following pieces of Context to answer the question at the end.
    #         If you don't know the answer, just say that you don't know, don't try to make up an answer.
    #         Use the EXACT TEXT from the Context WITHOUT ANY MODIFICATIONS, REORGANIZATION or EMBELLISHMENT.
    #         Don't try to answer anything that isn't in Context.
    #         Context:
    #         ```
    #         {context}
    #         ```
    #         """
    system_text = f"""
次のコンテキストを使用して、最後にある質問に答えてください。  
もし答えがわからない場合は、「わかりません」と言ってください。答えをでっち上げようとしないでください。  
コンテキストの正確なテキストを使用し、**一切の修正、再構成、または脚色を加えずに**使用してください。  
コンテキストにないことについては答えようとしないでください。  

コンテキスト:
```
{context}
``` \n"""

    user_text = f"""
            質問: 
            ```
            {query_text}
            ```
            役に立つ回答: \n
    """

    command_r_user_text = user_text
    command_r_plus_user_text = user_text
    openai_gpt4o_user_text = user_text
    openai_gpt4_user_text = user_text
    claude_3_opus_user_text = user_text
    claude_3_sonnet_user_text = user_text
    claude_3_haiku_user_text = user_text

    async for r, r_plus, gpt4o, gpt4, opus, sonnet, haiku in chat(system_text, command_r_user_text,
                                                                  command_r_plus_user_text,
                                                                  openai_gpt4o_user_text, openai_gpt4_user_text,
                                                                  claude_3_opus_user_text, claude_3_sonnet_user_text,
                                                                  claude_3_haiku_user_text, command_r_checkbox,
                                                                  command_r_plus_checkbox, openai_gpt4o_checkbox,
                                                                  openai_gpt4_checkbox, claude_3_opus_checkbox,
                                                                  claude_3_sonnet_checkbox, claude_3_haiku_checkbox):
        command_r_response += r
        command_r_plus_response += r_plus
        openai_gpt4o_response += gpt4o
        openai_gpt4_response += gpt4
        claude_3_opus_response += opus
        claude_3_sonnet_response += sonnet
        claude_3_haiku_response += haiku
        yield (command_r_response, command_r_plus_response, openai_gpt4o_response, openai_gpt4_response,
               claude_3_opus_response, claude_3_sonnet_response, claude_3_haiku_response)


async def eval_ragas(llm_answer_checkbox,
                     llm_evaluation_checkbox,
                     system_text, standard_answer_text,
                     command_r_response, command_r_plus_response, openai_gpt4o_response,
                     openai_gpt4_response, claude_3_opus_response, claude_3_sonnet_response, claude_3_haiku_response):
    def remove_last_line(text):
        if text:
            lines = text.splitlines()
            if lines[-1].startswith("推論時間"):
                lines.pop()
            return '\n'.join(lines)
        else:
            return text

    if standard_answer_text:
        standard_answer_text = standard_answer_text.strip()
    else:
        standard_answer_text = "入力されていません。"
    print(f"{llm_evaluation_checkbox=}")
    if not llm_evaluation_checkbox:
        yield ("", "", "", "", "", "", "")
    else:
        command_r_checkbox = False
        command_r_plus_checkbox = False
        openai_gpt4o_checkbox = False
        openai_gpt4_checkbox = False
        claude_3_opus_checkbox = False
        claude_3_sonnet_checkbox = False
        claude_3_haiku_checkbox = False
        if "cohere/command-r" in llm_answer_checkbox:
            command_r_checkbox = True
        if "cohere/command-r-plus" in llm_answer_checkbox:
            command_r_plus_checkbox = True
        if "openai/gpt-4o" in llm_answer_checkbox:
            openai_gpt4o_checkbox = True
        if "openai/gpt-4" in llm_answer_checkbox:
            openai_gpt4_checkbox = True
        if "claude/opus" in llm_answer_checkbox:
            claude_3_opus_checkbox = True
        if "claude/sonnet" in llm_answer_checkbox:
            claude_3_sonnet_checkbox = True
        if "claude/haiku" in llm_answer_checkbox:
            claude_3_haiku_checkbox = True

        command_r_response = remove_last_line(command_r_response)
        command_r_plus_response = remove_last_line(command_r_plus_response)
        openai_gpt4o_response = remove_last_line(openai_gpt4o_response)
        openai_gpt4_response = remove_last_line(openai_gpt4_response)
        claude_3_opus_response = remove_last_line(claude_3_opus_response)
        claude_3_sonnet_response = remove_last_line(claude_3_sonnet_response)
        claude_3_haiku_response = remove_last_line(claude_3_haiku_response)

        command_r_user_text = f"""
-標準回答-
 {standard_answer_text}

-与えられた回答-
 {command_r_response}

-出力-\n　"""

        command_r_plus_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{command_r_plus_response}

-出力-\n　"""

        openai_gpt4o_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{openai_gpt4o_response}

-出力-\n　"""

        openai_gpt4_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{openai_gpt4_response}

-出力-\n　"""

        claude_3_opus_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{claude_3_opus_response}

-出力-\n　"""

        claude_3_sonnet_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{claude_3_sonnet_response}

-出力-\n　"""

        claude_3_haiku_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{claude_3_haiku_response}

-出力-\n　"""

        eval_command_r_response = ""
        eval_command_r_plus_response = ""
        eval_openai_gpt4o_response = ""
        eval_openai_gpt4_response = ""
        eval_claude_3_opus_response = ""
        eval_claude_3_sonnet_response = ""
        eval_claude_3_haiku_response = ""

        async for r, r_plus, gpt4o, gpt4, opus, sonnet, haiku in chat(system_text, command_r_user_text,
                                                                      command_r_plus_user_text,
                                                                      openai_gpt4o_user_text, openai_gpt4_user_text,
                                                                      claude_3_opus_user_text,
                                                                      claude_3_sonnet_user_text,
                                                                      claude_3_haiku_user_text, command_r_checkbox,
                                                                      command_r_plus_checkbox, openai_gpt4o_checkbox,
                                                                      openai_gpt4_checkbox, claude_3_opus_checkbox,
                                                                      claude_3_sonnet_checkbox,
                                                                      claude_3_haiku_checkbox):
            eval_command_r_response += r
            eval_command_r_plus_response += r_plus
            eval_openai_gpt4o_response += gpt4o
            eval_openai_gpt4_response += gpt4
            eval_claude_3_opus_response += opus
            eval_claude_3_sonnet_response += sonnet
            eval_claude_3_haiku_response += haiku
            yield (eval_command_r_response, eval_command_r_plus_response, eval_openai_gpt4o_response,
                   eval_openai_gpt4_response,
                   eval_claude_3_opus_response, eval_claude_3_sonnet_response, eval_claude_3_haiku_response)


def generate_download_file(search_result, llm_answer_checkbox, llm_evaluation_checkbox,
                           query_text, standard_answer_text,
                           command_r_response, command_r_plus_response, openai_gpt4o_response,
                           openai_gpt4_response, claude_3_opus_response, claude_3_sonnet_response,
                           claude_3_haiku_response,
                           command_r_evaluation, command_r_plus_evaluation, openai_gpt4o_evaluation,
                           openai_gpt4_evaluation, claude_3_opus_evaluation, claude_3_sonnet_evaluation,
                           claude_3_haiku_evaluation
                           ):
    # 创建一些示例 DataFrame
    if llm_evaluation_checkbox:
        standard_answer_text = standard_answer_text
    else:
        standard_answer_text = ""
    df1 = pd.DataFrame({'クエリ': [query_text], '標準回答': [standard_answer_text]})

    df2 = search_result

    if "cohere/command-r" in llm_answer_checkbox:
        command_r_response = command_r_response
        if llm_evaluation_checkbox:
            command_r_evaluation = command_r_evaluation
        else:
            command_r_evaluation = ""
    else:
        command_r_response = ""
        command_r_evaluation = ""
    if "cohere/command-r-plus" in llm_answer_checkbox:
        command_r_plus_response = command_r_plus_response
        if llm_evaluation_checkbox:
            command_r_plus_evaluation = command_r_plus_evaluation
        else:
            command_r_plus_evaluation = ""
    else:
        command_r_plus_response = ""
        command_r_plus_evaluation = ""
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt4o_response = openai_gpt4o_response
        if llm_evaluation_checkbox:
            openai_gpt4o_evaluation = openai_gpt4o_evaluation
        else:
            openai_gpt4o_evaluation = ""
    else:
        openai_gpt4o_response = ""
        openai_gpt4o_evaluation = ""
    if "openai/gpt-4" in llm_answer_checkbox:
        openai_gpt4_response = openai_gpt4_response
        if llm_evaluation_checkbox:
            openai_gpt4_evaluation = openai_gpt4_evaluation
        else:
            openai_gpt4_evaluation = ""
    else:
        openai_gpt4_response = ""
        openai_gpt4_evaluation = ""
    if "claude/opus" in llm_answer_checkbox:
        claude_3_opus_response = claude_3_opus_response
        if llm_evaluation_checkbox:
            claude_3_opus_evaluation = claude_3_opus_evaluation
        else:
            claude_3_opus_evaluation = ""
    else:
        claude_3_opus_response = ""
        claude_3_opus_evaluation = ""
    if "claude/sonnet" in llm_answer_checkbox:
        claude_3_sonnet_response = claude_3_sonnet_response
        if llm_evaluation_checkbox:
            claude_3_sonnet_evaluation = claude_3_sonnet_evaluation
        else:
            claude_3_sonnet_evaluation = ""
    else:
        claude_3_sonnet_response = ""
        claude_3_sonnet_evaluation = ""
    if "claude/haiku" in llm_answer_checkbox:
        claude_3_haiku_response = claude_3_haiku_response
        if llm_evaluation_checkbox:
            claude_3_haiku_evaluation = claude_3_haiku_evaluation
        else:
            claude_3_haiku_evaluation = ""
    else:
        claude_3_haiku_response = ""
        claude_3_haiku_evaluation = ""

    df3 = pd.DataFrame({'LLM モデル': ["cohere/command-r", "cohere/command-r-plus", "openai/gpt-4o", "openai/gpt-4",
                                       "claude/opus", "claude/sonnet", "claude/haiku"],
                        'LLM メッセージ': [command_r_response, command_r_plus_response, openai_gpt4o_response,
                                           openai_gpt4_response, claude_3_opus_response, claude_3_sonnet_response,
                                           claude_3_haiku_response],
                        'LLM 評価': [command_r_evaluation, command_r_plus_evaluation, openai_gpt4o_evaluation,
                                     openai_gpt4_evaluation, claude_3_opus_evaluation, claude_3_sonnet_evaluation,
                                     claude_3_haiku_evaluation]})

    # 定义文件路径
    filepath = '/tmp/result.xlsx'

    # 使用 ExcelWriter 将多个 DataFrame 写入不同的 sheet
    with pd.ExcelWriter(filepath) as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
        df3.to_excel(writer, sheet_name='Sheet3', index=False)

    print(f"Excel 文件已保存到 {filepath}")
    return gr.DownloadButton(value=filepath, visible=True)


def delete_document(server_directory, doc_ids):
    if not server_directory:
        raise gr.Error("サーバー・ディレクトリを入力してください")
    print(f"{doc_ids=}")
    if not doc_ids or len(doc_ids) == 0 or (len(doc_ids) == 1 and doc_ids[0] == ''):
        raise gr.Error("ドキュメントを選択してください")

    output_sql = ""
    with pool.acquire() as conn, conn.cursor() as cursor:
        for doc_id in filter(bool, doc_ids):
            server_path = get_server_path(doc_id)
            if os.path.exists(server_path):
                os.remove(server_path)
                print(f"File {doc_id} deleted successfully")
            else:
                print(f"File {doc_id} not found")

            delete_embedding_sql = f"""
DELETE FROM {DEFAULT_COLLECTION_NAME}_embedding
WHERE doc_id = :doc_id """
            delete_collection_sql = f"""
DELETE FROM {DEFAULT_COLLECTION_NAME}_collection
WHERE id = :doc_id """
            output_sql += delete_embedding_sql.strip().replace(":doc_id", "'" + doc_id + "'") + "\n"
            output_sql += delete_collection_sql.strip().replace(":doc_id", "'" + doc_id + "'")
            cursor.execute(delete_embedding_sql, doc_id=doc_id)
            cursor.execute(delete_collection_sql, doc_id=doc_id)

            conn.commit()

    doc_list = get_doc_list()
    return gr.Textbox(value=output_sql), gr.Radio(doc_list), gr.CheckboxGroup(choices=doc_list,
                                                                              value=""), gr.CheckboxGroup(
        choices=doc_list)


with gr.Blocks(css=custom_css) as app:
    gr.Markdown(value="# RAG精度あげたろう", elem_classes="main_Header")
    gr.Markdown(value="### LLM&RAG精度検証ツール", elem_classes="sub_Header")
    with gr.Tabs() as tabs:
        with gr.TabItem(label="環境設定") as tab_setting:
            with gr.TabItem(label="OCI GenAIの設定*") as tab_create_oci_cred:
                with gr.Accordion(label="使用されたSQL", open=False) as tab_create_oci_cred_sql_accordion:
                    tab_create_oci_cred_sql_text = gr.Textbox(label="SQL", show_label=False, lines=25, max_lines=50,
                                                              autoscroll=False, interactive=False,
                                                              show_copy_button=True)
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_user_ocid_text = gr.Textbox(label="User OCID*", lines=1, interactive=True)
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_tenancy_ocid_text = gr.Textbox(label="Tenancy OCID*", lines=1,
                                                                           interactive=True)
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_fingerprint_text = gr.Textbox(label="Fingerprint*", lines=1,
                                                                          interactive=True)
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_private_key_file = gr.File(label="Private Key*", file_types=[".pem"],
                                                                       type="filepath", interactive=True)
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_clear_button = gr.ClearButton(value="クリア")
                    with gr.Column():
                        tab_create_oci_cred_button = gr.Button(value="設定/再設定", variant="primary")
                with gr.Accordion(label="OCI GenAIのテスト", open=False) as tab_create_oci_cred_test_accordion:
                    with gr.Row():
                        with gr.Column():
                            tab_create_oci_cred_test_query_text = gr.Textbox(label="テキスト", lines=1, max_lines=1,
                                                                             value="こんにちわ")
                    with gr.Row():
                        with gr.Column():
                            tab_create_oci_cred_test_vector_text = gr.Textbox(label="ベクトル", lines=10, max_lines=10,
                                                                              autoscroll=False)
                    with gr.Row():
                        with gr.Column():
                            tab_create_oci_cred_test_button = gr.Button(value="テスト", variant="primary")
            with gr.TabItem(label="テーブルの作成*") as tab_create_table:
                with gr.Accordion(label="使用されたSQL", open=False) as tab_create_table_sql_accordion:
                    tab_create_table_sql_text = gr.Textbox(label="SQL", show_label=False, lines=25, max_lines=50,
                                                           autoscroll=False, interactive=False,
                                                           show_copy_button=True)
                with gr.Row():
                    with gr.Column():
                        tab_create_table_button = gr.Button(value="作成/再作成", variant="primary")
            with gr.TabItem(label="Cohereの設定(オプション)") as tab_create_cohere_cred:
                with gr.Row():
                    with gr.Column():
                        tab_create_cohere_cred_api_key_text = gr.Textbox(label="API Key*", lines=1, interactive=True)
                with gr.Row():
                    with gr.Column():
                        tab_create_cohere_cred_button = gr.Button(value="設定/再設定", variant="primary")
            with gr.TabItem(label="OpenAIの設定(オプション)") as tab_create_openai_cred:
                with gr.Row():
                    with gr.Column():
                        tab_create_openai_cred_base_url_text = gr.Textbox(label="Base URL*", lines=1, interactive=True)
                with gr.Row():
                    with gr.Column():
                        tab_create_openai_cred_api_key_text = gr.Textbox(label="API Key*", lines=1, interactive=True)
                with gr.Row():
                    with gr.Column():
                        tab_create_openai_cred_button = gr.Button(value="設定/再設定", variant="primary")
            with gr.TabItem(label="Claudeの設定(オプション)") as tab_create_claude_cred:
                with gr.Row():
                    with gr.Column():
                        tab_create_claude_cred_api_key_text = gr.Textbox(label="API Key*", lines=1, interactive=True)
                with gr.Row():
                    with gr.Column():
                        tab_create_claude_cred_button = gr.Button(value="設定/再設定", variant="primary")
            with gr.TabItem(label="Langfuseの設定(オプション)") as tab_create_langfuse_cred:
                with gr.Row():
                    with gr.Column():
                        tab_create_langfuse_cred_secret_key_text = gr.Textbox(label="LANGFUSE_SECRET_KEY*", lines=1,
                                                                              interactive=True, placeholder="sk-lf-...")
                with gr.Row():
                    with gr.Column():
                        tab_create_langfuse_cred_public_key_text = gr.Textbox(label="LANGFUSE_PUBLIC_KEY*", lines=1,
                                                                              interactive=True, placeholder="pk-lf-...")
                with gr.Row():
                    with gr.Column():
                        tab_create_langfuse_cred_host_text = gr.Textbox(label="LANGFUSE_HOST*", lines=1,
                                                                        interactive=True,
                                                                        placeholder="http://xxx.xxx.xxx.xxx:3000")
                with gr.Row():
                    with gr.Column():
                        tab_create_langfuse_cred_button = gr.Button(value="設定/再設定", variant="primary")
        with gr.TabItem(label="LLM評価") as tab_llm_evaluation:
            with gr.TabItem(label="LLMとチャット") as tab_chat_with_llm:
                with gr.Row():
                    with gr.Column():
                        tab_chat_with_llm_answer_checkbox_group = gr.CheckboxGroup(
                            ["cohere/command-r", "cohere/command-r-plus", "openai/gpt-4o", "openai/gpt-4",
                             "claude/opus", "claude/sonnet", "claude/haiku"],
                            label="LLM モデル*", value=[])
                with gr.Accordion(label="Command-R メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_with_llm_command_r_accordion:
                    tab_chat_with_command_r_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                     lines=2,
                                                                     autoscroll=True, interactive=False,
                                                                     show_copy_button=True)
                with gr.Accordion(label="Command-R+ メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_with_llm_command_r_plus_accordion:
                    tab_chat_with_command_r_plus_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                          lines=2,
                                                                          autoscroll=True, interactive=False,
                                                                          show_copy_button=True)
                with gr.Accordion(label="OpenAI gpt-4o メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_with_llm_openai_gpt4o_accordion:
                    tab_chat_with_openai_gpt4o_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                        lines=2,
                                                                        autoscroll=True, interactive=False,
                                                                        show_copy_button=True)
                with gr.Accordion(label="OpenAI gpt-4 メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_with_llm_openai_gpt4_accordion:
                    tab_chat_with_openai_gpt4_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                       lines=2,
                                                                       autoscroll=True, interactive=False,
                                                                       show_copy_button=True)
                with gr.Accordion(label="Claude 3 Opus メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_with_llm_claude_3_opus_accordion:
                    tab_chat_with_claude_3_opus_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                         lines=2,
                                                                         autoscroll=True, interactive=False,
                                                                         show_copy_button=True)
                with gr.Accordion(label="Claude 3.5 Sonnet メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_with_llm_claude_3_sonnet_accordion:
                    tab_chat_with_claude_3_sonnet_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                           lines=2,
                                                                           autoscroll=True, interactive=False,
                                                                           show_copy_button=True)
                with gr.Accordion(label="Claude 3 Haiku メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_with_llm_claude_3_haiku_accordion:
                    tab_chat_with_claude_3_haiku_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                          lines=2,
                                                                          autoscroll=True, interactive=False,
                                                                          show_copy_button=True)
                with gr.Accordion(open=False, label="システム・メッセージ"):
                    #                     tab_chat_with_llm_system_text = gr.Textbox(label="システム・メッセージ*", show_label=False, lines=5,
                    #                                                                max_lines=15,
                    #                                                                value="You are a helpful assistant. \n\
                    # Please respond to me in the same language I use for my messages. \n\
                    # If I switch languages, please switch your responses accordingly.")
                    tab_chat_with_llm_system_text = gr.Textbox(label="システム・メッセージ*", show_label=False, lines=5,
                                                               max_lines=15,
                                                               value="""あなたは役立つアシスタントです。
私のメッセージと同じ言語で返答してください。
もし私が言語を切り替えた場合は、それに応じて返答の言語も切り替えてください。""")
                with gr.Row():
                    with gr.Column():
                        tab_chat_with_llm_query_text = gr.Textbox(label="ユーザー・メッセージ*", lines=2, max_lines=5)
                with gr.Row():
                    with gr.Column():
                        tab_chat_with_llm_clear_button = gr.ClearButton(value="クリア")
                    with gr.Column():
                        tab_chat_with_llm_chat_button = gr.Button(value="送信", variant="primary")
                with gr.Row(visible=False):
                    with gr.Column():
                        gr.Examples(examples=[], inputs=tab_chat_with_llm_query_text)
        with gr.TabItem(label="RAG評価", elem_classes="inner_tab") as tab_rag_evaluation:
            with gr.TabItem(label="Step-1.読込み") as tab_load_document:
                with gr.Accordion(label="使用されたSQL", open=False) as tab_load_document_sql_accordion:
                    tab_load_document_output_sql_text = gr.Textbox(label="使用されたSQL", show_label=False, lines=10,
                                                                   autoscroll=False,
                                                                   show_copy_button=True)
                with gr.Row():
                    with gr.Column():
                        tab_load_document_doc_id_text = gr.Textbox(label="Doc ID", lines=1, interactive=False)
                with gr.Row(visible=False):
                    with gr.Column():
                        tab_load_document_page_count_text = gr.Textbox(label="ページ数", lines=1, interactive=False)
                with gr.Row():
                    with gr.Column():
                        tab_load_document_page_content_text = gr.Textbox(label="コンテンツ", lines=15, max_lines=15,
                                                                         autoscroll=False,
                                                                         show_copy_button=True, interactive=False)
                with gr.Row():
                    with gr.Column():
                        tab_load_document_file_text = gr.File(label="ファイル*",
                                                              file_types=["txt", "csv", "doc", "docx", "epub", "image",
                                                                          "md", "msg", "odt", "org", "pdf", "ppt",
                                                                          "pptx",
                                                                          "rtf", "rst", "tsv", "xls", "xlsx"],
                                                              type="filepath")
                    with gr.Column():
                        tab_load_document_server_directory_text = gr.Text(label="サーバー・ディレクトリ*",
                                                                          value="/u01/data/no1rag/")
                with gr.Row(visible=False):
                    with gr.Column():
                        gr.Examples(examples=[],
                                    label="サンプル・ファイル",
                                    inputs=tab_load_document_file_text)
                with gr.Row():
                    with gr.Column():
                        tab_load_document_load_button = gr.Button(value="読込み", variant="primary")
            with gr.TabItem(label="Step-2.分割・ベクトル化・保存") as tab_split_document:
                with gr.Accordion(label="使用されたSQL", open=False) as tab_split_document_sql_accordion:
                    tab_split_document_output_sql_text = gr.Textbox(label="使用されたSQL", show_label=False, lines=10,
                                                                    autoscroll=False,
                                                                    show_copy_button=True)
                with gr.Row():
                    tab_split_document_chunks_count = gr.Textbox(label="チャンク数", lines=1)
                with gr.Row():
                    tab_split_document_chunks_result_dataframe = gr.Dataframe(
                        label="チャンク結果",
                        headers=["CHUNK_ID", "CHUNK_OFFSET", "CHUNK_LENGTH", "CHUNK_DATA"],
                        datatype=["str", "str", "str", "str"],
                        row_count=20,
                        col_count=(4, "fixed"),
                        wrap=True,
                        column_widths=["10%", "10%", "10%", "70%"]
                    )
                with gr.Row():
                    with gr.Column():
                        # doc_id_text = gr.Textbox(label="Doc ID*", lines=1)
                        tab_split_document_doc_id_radio = gr.Radio(
                            choices=get_doc_list(),
                            label="ドキュメント*"
                        )
                # with gr.Accordion("UTL_TO_CHUNKS 設定*"):
                with gr.Accordion("CHUNKS 設定*"):
                    with gr.Row():
                        with gr.Column():
                            tab_split_document_chunks_language_radio = gr.Radio(label="LANGUAGE",
                                                                                choices=[("JAPANESE", "JAPANESE"),
                                                                                         ("AMERICAN", "AMERICAN")],
                                                                                value="JAPANESE",
                                                                                visible=False,
                                                                                info="Default value: JAPANESE。入力テキストの言語を指定。テキストに言語によって解釈が異なる可能性のある特定の文字(例えば、句読点や略語)が含まれる場合、この言語の指定は特に重要です。Oracle Database Globalization Support Guideに記載されているNLSでサポートされている言語名または略称を指定できる。")
                        with gr.Column():
                            tab_split_document_chunks_by_radio = gr.Radio(label="BY",
                                                                          choices=[("BY CHARACTERS (or BY CHARS )",
                                                                                    "CHARACTERS"),
                                                                                   ("BY WORDS", "WORDS"),
                                                                                   ("BY VOCABULARY", "VOCABULARY")],
                                                                          value="WORDS",
                                                                          visible=False,
                                                                          info="Default value: BY WORDS。データ分割の方法を文字、単語、語彙トークンで指定。BY CHARACTERS: 文字数で計算して分割。BY WORDS: 単語数を計算して分割、単語ごとに空白文字が入る言語が対象、日本語、中国語、タイ語などの場合、各ネイティブ文字は単語（ユニグラム）としてみなされる。BY VOCABULARY: 語彙のトークン数を計算して分割、CREATE_VOCABULARYパッケージを使って、語彙登録が可能。", )
                    with gr.Row():
                        with gr.Column():
                            # tab_split_document_chunks_max_text = gr.Text(label="Max",
                            #                                              value="256",
                            #                                              lines=1,
                            #                                              info="",
                            #                                              )
                            tab_split_document_chunks_max_slider = gr.Slider(label="Max",
                                                                             value=256,
                                                                             minimum=64,
                                                                             maximum=384,
                                                                             step=1,
                                                                             )
                        with gr.Column():
                            tab_split_document_chunks_overlap_slider = gr.Slider(label="Overlap(Percentage of Max)",
                                                                                 minimum=0,
                                                                                 maximum=20,
                                                                                 step=5,
                                                                                 value=10,
                                                                                 info="",
                                                                                 )
                    with gr.Row():
                        with gr.Column():
                            tab_split_document_chunks_split_by_radio = gr.Radio(label="SPLIT [BY]",
                                                                                choices=[("NONE", "NONE"),
                                                                                         ("NEWLINE", "NEWLINE"),
                                                                                         ("BLANKLINE", "BLANKLINE"),
                                                                                         ("SPACE", "SPACE"),
                                                                                         ("RECURSIVELY", "RECURSIVELY"),
                                                                                         ("SENTENCE", "SENTENCE"),
                                                                                         ("CUSTOM", "CUSTOM")],
                                                                                value="RECURSIVELY",
                                                                                visible=False,
                                                                                info="Default value: RECURSIVELY。テキストが最大サイズに達したときに、どうやって分割するかを指定。チャンクの適切な境界を定義するために使用する。NONE: MAX指定されている文字数、単語数、語彙トークン数に達したら分割。NEWLINE: MAX指定サイズを超えてテキストの行末で分割。BLANKLINE: 指定サイズを超えてBLANKLINE（2回の改行）の末尾で分割。SPACE: MAX指定サイズを超えて空白の行末で分割。RECURSIVELY: BLANKLINE, NEWLINE, SPACE, NONEの順に条件に応じて分割する。1.入力テキストがmax値以上の場合、最初の分割文字で分割、2.1.が失敗した場合に、2番目の分割文字で分割、3.分割文字が存在しない場合、テキスト中のどの位置においてもMAXで分割。SENTENCE: 文末の句読点で分割BY WORDSとBY VOCABULARYでのみ指定可能。MAX設定の仕方によっては必ず句読点で区切られるわけではないので注意。例えば、文がMAX値よりも大きい場合、MAX値で区切られる。MAX値よりも小さな文の場合で、2文以上がMAXの制限内に収まるのであれば1つに収める。CUSTOM: カスタム分割文字リストに基づいて分割、分割文字は最大16個まで、長さはそれぞれ10文字までで指定可能。")
                        with gr.Column():
                            tab_split_document_chunks_split_by_custom_text = gr.Text(
                                label="CUSTOM SPLIT CHARACTERS(SPLIT [BY] = CUSTOMの場合のみ有効)",
                                # value="'\u3002', '.'",
                                visible=False,
                                info="カンマ区切りのカスタム分割文字リストに基づいて分割、分割文字は最大16個まで、長さはそれぞれ10文字までで指定可能。タブ (\t)、改行 (\n)、およびラインフィード (\r) のシーケンスのみを省略できる。たとえば、'<html>','</html>'")
                    with gr.Row():
                        with gr.Column():
                            tab_split_document_chunks_normalize_radio = gr.Radio(label="NORMALIZE",
                                                                                 choices=[("NONE", "NONE"),
                                                                                          ("ALL", "ALL"),
                                                                                          ("OPTIONS", "OPTIONS")],
                                                                                 value="ALL",
                                                                                 visible=False,
                                                                                 info="Default value: ALL。ドキュメントをテキストに変換する際にありがちな問題について、自動的に前処理、後処理を実行し高品質なチャンクとして格納するために使用。NONE: 処理を行わない。ALL: マルチバイトの句読点をシングルバイトに正規化。OPTIONS: PUNCTUATION: スマート引用符、スマートハイフン、マルチバイト句読点をテキストに含める。WHITESPACE: 不要な文字を削除して空白を最小限に抑える例えば空白行はそのままに、余分な改行やスペース、タブを削除する。WIDECHAR: マルチバイト数字とローマ字をシングルバイトに正規化する")
                        with gr.Column():
                            tab_split_document_chunks_normalize_options_checkbox_group = gr.CheckboxGroup(
                                label="NORMALIZE OPTIONS(NORMALIZE = OPTIONSの場合のみ有効かつ必須)",
                                choices=[
                                    ("PUNCTUATION", "PUNCTUATION"),
                                    ("WHITESPACE", "WHITESPACE"),
                                    ("WIDECHAR", "WIDECHAR")],
                                visible=False,
                                info="ドキュメントをテキストに変換する際にありがちな問題について、自動的に前処理、後処理を実行し高品質なチャンクとして格納するために使用。PUNCTUATION: スマート引用符、スマートハイフン、マルチバイト句読点をテキストに含める。WHITESPACE: 不要な文字を削除して空白を最小限に抑える例えば空白行はそのままに、余分な改行やスペース、タブを削除する。WIDECHAR: マルチバイト数字とローマ字をシングルバイトに正規化する")

                with gr.Row():
                    with gr.Column():
                        tab_split_document_split_button = gr.Button(value="分割", variant="primary")
                    with gr.Column():
                        tab_split_document_embed_save_button = gr.Button(
                            value="ベクトル化して保存（データ量が多いと時間がかかる）",
                            variant="primary")
            with gr.TabItem(label="Step-3.削除(オプション)") as tab_delete_document:
                with gr.Accordion(label="使用されたSQL", open=False) as tab_delete_document_sql_accordion:
                    tab_delete_document_delete_sql = gr.Textbox(label="生成されたSQL", show_label=False, lines=10,
                                                                autoscroll=False, show_copy_button=True)
                with gr.Row():
                    with gr.Column():
                        tab_delete_document_server_directory_text = gr.Text(label="サーバー・ディレクトリ*",
                                                                            value="/u01/data/no1rag/")
                with gr.Row():
                    with gr.Column():
                        # doc_id_text = gr.Textbox(label="Doc ID*", lines=1)
                        tab_delete_document_doc_ids_checkbox_group = gr.CheckboxGroup(
                            choices=get_doc_list(),
                            label="ドキュメント*"
                        )
                with gr.Row():
                    with gr.Column():
                        tab_delete_document_delete_button = gr.Button(value="削除", variant="primary")
            with gr.TabItem(label="Step-4.ドキュメントとチャット") as tab_chat_document:
                with gr.Row():
                    with gr.Column():
                        tab_chat_document_llm_answer_checkbox_group = gr.CheckboxGroup(
                            ["cohere/command-r", "cohere/command-r-plus", "openai/gpt-4o", "openai/gpt-4",
                             "claude/opus",
                             "claude/sonnet", "claude/haiku"],
                            label="LLM モデル", value=[""])
                with gr.Row():
                    with gr.Column():
                        tab_chat_document_question_embedding_model_checkbox_group = gr.CheckboxGroup(
                            ["cohere/embed-multilingual-v3.0"],
                            label="Embedding モデル*", value="cohere/embed-multilingual-v3.0", interactive=False)
                    with gr.Column():
                        tab_chat_document_reranker_model_radio = gr.Radio(
                            ["None", "cohere/rerank-multilingual-v3.0",
                             "cohere/rerank-english-v3.0"
                             ],
                            label="Rerank モデル*", value="None")
                with gr.Row():
                    with gr.Column():
                        tab_chat_document_top_k_slider = gr.Slider(label="類似検索Top-K*", minimum=1,
                                                                   maximum=100, step=1,
                                                                   info="Default value: 25。類似度距離の低い順（=類似度の高い順）で上位K件のみを抽出する。",
                                                                   interactive=True, value=25)
                    with gr.Column():
                        tab_chat_document_threshold_value_slider = gr.Slider(label="類似検索閾値*",
                                                                             minimum=0.10,
                                                                             info="Default value: 0.55。類似度距離が閾値以下のデータのみを抽出する。",
                                                                             maximum=0.95, step=0.05, value=0.55)
                with gr.Row():
                    with gr.Column():
                        tab_chat_document_reranker_top_k_slider = gr.Slider(label="Rerank Top-K*", minimum=1,
                                                                            maximum=50, step=1,
                                                                            info="Default value: 5。Rerank Scoreの高い順で上位K件のみを抽出する。",
                                                                            interactive=True, value=5)
                    with gr.Column():
                        tab_chat_document_reranker_threshold_slider = gr.Slider(label="Rerank Score閾値*",
                                                                                minimum=0.0,
                                                                                info="Default value: 0.4。Rerank Scoreが閾値以上のデータのみを抽出する。",
                                                                                maximum=0.99, step=0.001, value=0.40,
                                                                                interactive=True)
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_partition_by_k_slider = gr.Slider(label="Partition-By-K", minimum=0,
                                                                                maximum=20, step=1,
                                                                                interactive=True,
                                                                                info="Default value: 0。類似検索の対象ドキュメント数を指定。0: 全部。1: 1個。2：2個。... n: n個。")
                        with gr.Column():
                            tab_chat_document_answer_by_one_checkbox = gr.Checkbox(
                                label="Highest-Ranked-One 文書による回答",
                                value=False,
                                info="他のすべての文書を無視し、最上位にランクされた1つの文書のみによって回答する。")
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_extend_first_chunk_size = gr.Slider(label="Extend-First-K", minimum=0,
                                                                                  maximum=50, step=1,
                                                                                  interactive=True,
                                                                                  value=0,
                                                                                  info="Default value: 0。DISTANCE計算対象外。検索されたチャンクを拡張する数を指定。0: 拡張しない。 1: 最初の1個を拡張。2: 最初の2個を拡張。 ... n: 最初のn個を拡張。")
                        with gr.Column():
                            tab_chat_document_extend_around_chunk_size = gr.Slider(label="Extend-Around-K", minimum=0,
                                                                                   maximum=50, step=2,
                                                                                   interactive=True,
                                                                                   value=2,
                                                                                   info="Default value: 2。DISTANCE計算対象外。検索されたチャンクを拡張する数を指定。0: 拡張しない。 2: 2個で前後それぞれ1個を拡張。4: 4個で前後それぞれ2個を拡張。... n: n個で前後それぞれn/2個を拡張。")
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_text_search_checkbox = gr.Checkbox(label="テキスト検索", value=False,
                                                                                 info="テキスト検索は元のクエリに限定され、クエリの拡張で生成されたクエリは無視される。")
                        with gr.Column():
                            tab_chat_document_text_search_k_slider = gr.Slider(label="テキスト検索Limit-K",
                                                                               minimum=1,
                                                                               maximum=10, step=1,
                                                                               value=6,
                                                                               info="Default value: 6。テキスト検索に使用できる単語数の制限。")
                with gr.Row(visible=False):
                    tab_chat_document_accuracy_plan_radio = gr.Radio(
                        ["Somewhat Inaccurate", "Decent Accuracy", "Extremely Precise"],
                        label="Accuracy Plan*", value="Somewhat Inaccurate", interactive=True)
                with gr.Row():
                    tab_chat_document_llm_evaluation_checkbox = gr.Checkbox(label="評価", show_label=True,
                                                                            interactive=True, value=False)
                with gr.Row():
                    #                     tab_chat_document_system_message_text = gr.Textbox(label="システム・メッセージ*", lines=15,
                    #                                                                        max_lines=20,
                    #                                                                        interactive=True,
                    #                                                                        visible=False,
                    #                                                                        value=f"""You are an ANSWER EVALUATOR.
                    # Your task is to compare a given answer to a standard answer and evaluate its quality.
                    # Respond with a score from 0 to 10 for each of the following criteria:
                    # 1. Correctness (0 being completely incorrect, 10 being perfectly correct)
                    # 2. Completeness (0 being entirely incomplete, 10 being fully complete)
                    # 3. Clarity (0 being very unclear, 10 being extremely clear)
                    # 4. Conciseness (0 being extremely verbose, 10 being optimally concise)
                    #
                    # After providing scores, give a brief explanation for each score.
                    # Finally, provide an overall score from 0 to 10 and a summary of the evaluation.
                    # Please respond to me in the same language I use for my messages.
                    # If I switch languages, please switch your responses accordingly.
                    #                 """)
                    tab_chat_document_system_message_text = gr.Textbox(label="システム・メッセージ*", lines=15,
                                                                       max_lines=20,
                                                                       interactive=True,
                                                                       visible=False,
                                                                       value=f"""
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
もし私が言語を切り替えた場合は、それに応じて返答の言語も切り替えてください。\n""")
                with gr.Row():
                    tab_chat_document_standard_answer_text = gr.Textbox(label="標準回答*", lines=2, interactive=True,
                                                                        visible=False)
                with gr.Accordion("ドキュメント*", open=True):
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_doc_id_all_checkbox = gr.Checkbox(label="全部", value=True)
                    with gr.Row():
                        with gr.Column():
                            # doc_id_text = gr.Textbox(label="Doc ID*", lines=1)
                            tab_chat_document_doc_id_checkbox_group = gr.CheckboxGroup(
                                choices=get_doc_list(),
                                label="ドキュメント*",
                                show_label=False,
                                interactive=False
                            )
                with gr.Row() as tab_chat_document_searched_query_row:
                    with gr.Column():
                        tab_chat_document_query_text = gr.Textbox(label="クエリ*", lines=1)
                # with gr.Accordion("Sub-Query/RAG-Fusion/HyDE/Step-Back-Prompting/Customized-Multi-Step-Query", open=True):
                with gr.Accordion("クエリの拡張", open=False):
                    with gr.Row():
                        # generate_query_radio = gr.Radio(
                        #     ["None", "Sub-Query", "RAG-Fusion", "HyDE", "Step-Back-Prompting",
                        #      "Customized-Multi-Step-Query"],
                        tab_chat_document_generate_query_radio = gr.Radio(
                            [("None", "None"), ("サブクエリ", "Sub-Query"), ("類似クエリ", "RAG-Fusion"),
                             ("仮回答", "HyDE"), ("抽象化クエリ", "Step-Back-Prompting")],
                            label="LLMによって生成？", value="None", interactive=True)
                    with gr.Row():
                        tab_chat_document_sub_query1_text = gr.Textbox(
                            # label="(Sub-Query)サブクエリ1/(RAG-Fusion)類似クエリ1/(HyDE)仮回答1/(Step-Back-Prompting)抽象化クエリ1/(Customized-Multi-Step-Query)マルチステップクエリ1",
                            label="生成されたクエリ1",
                            lines=1, interactive=True, info="")
                    with gr.Row():
                        tab_chat_document_sub_query2_text = gr.Textbox(
                            # label="(Sub-Query)サブクエリ2/(RAG-Fusion)類似クエリ2/(HyDE)仮回答2/(Step-Back-Prompting)抽象化クエリ2/(Customized-Multi-Step-Query)マルチステップクエリ2",
                            label="生成されたクエリ2",
                            lines=1, interactive=True)
                    with gr.Row():
                        tab_chat_document_sub_query3_text = gr.Textbox(
                            # label="(Sub-Query)サブクエリ3/(RAG-Fusion)類似クエリ3/(HyDE)仮回答3/(Step-Back-Prompting)抽象化クエリ3/(Customized-Multi-Step-Query)マルチステップクエリ3",
                            label="生成されたクエリ3",
                            lines=1, interactive=True)
                with gr.Row() as tab_chat_document_chat_document_row:
                    with gr.Column():
                        tab_chat_document_chat_document_button = gr.Button(value="送信", variant="primary")
                with gr.Accordion(label="使用されたSQL", open=False) as tab_chat_document_sql_accordion:
                    tab_chat_document_output_sql_text = gr.Textbox(label="使用されたSQL", show_label=False, lines=25,
                                                                   max_lines=25, autoscroll=False,
                                                                   show_copy_button=True)
                with gr.Row() as tab_chat_document_searched_data_summary_row:
                    with gr.Column(scale=10):
                        tab_chat_document_searched_data_summary_text = gr.Markdown(value="", visible=False)
                    with gr.Column(scale=2):
                        tab_chat_document_download_output_button = gr.DownloadButton(label="ダウンロード",
                                                                                     visible=False,
                                                                                     variant="secondary", size="sm")
                with gr.Row() as searched_result_row:
                    with gr.Column():
                        tab_chat_document_searched_result_dataframe = gr.Dataframe(
                            headers=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"],
                            datatype=["str", "str", "str", "str", "str", "str", "str"],
                            row_count=10,
                            height=400,
                            col_count=(7, "fixed"),
                            wrap=True,
                            column_widths=["4%", "62%", "6%", "8%", "6%", "6%", "8%"],
                            interactive=False,
                        )
                with gr.Accordion(label="Command-R メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_document_llm_command_r_accordion:
                    tab_chat_document_command_r_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                         lines=2,
                                                                         autoscroll=True, interactive=False,
                                                                         show_copy_button=True)
                    with gr.Accordion(label="評価結果", visible=False,
                                      open=True) as tab_chat_document_llm_command_r_evaluation_accordion:
                        tab_chat_document_command_r_evaluation_text = gr.Textbox(label="評価結果", show_label=False,
                                                                                 lines=2,
                                                                                 autoscroll=True, interactive=False,
                                                                                 show_copy_button=True)
                with gr.Accordion(label="Command-R+ メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_document_llm_command_r_plus_accordion:
                    tab_chat_document_command_r_plus_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                              lines=2,
                                                                              autoscroll=True, interactive=False,
                                                                              show_copy_button=True)
                    with gr.Accordion(label="評価結果", visible=False,
                                      open=True) as tab_chat_document_llm_command_r_plus_evaluation_accordion:
                        tab_chat_document_command_r_plus_evaluation_text = gr.Textbox(label="評価結果",
                                                                                      show_label=False,
                                                                                      lines=2,
                                                                                      autoscroll=True,
                                                                                      interactive=False,
                                                                                      show_copy_button=True)
                with gr.Accordion(label="OpenAI gpt-4o メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_document_llm_openai_gpt4o_accordion:
                    tab_chat_document_openai_gpt4o_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                            lines=2,
                                                                            autoscroll=True, interactive=False,
                                                                            show_copy_button=True)
                    with gr.Accordion(label="評価結果", visible=False,
                                      open=True) as tab_chat_document_llm_openai_gpt4o_evaluation_accordion:
                        tab_chat_document_openai_gpt4o_evaluation_text = gr.Textbox(label="評価結果", show_label=False,
                                                                                    lines=2,
                                                                                    autoscroll=True, interactive=False,
                                                                                    show_copy_button=True)
                with gr.Accordion(label="OpenAI gpt-4 メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_document_llm_openai_gpt4_accordion:
                    tab_chat_document_openai_gpt4_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                           lines=2,
                                                                           autoscroll=True, interactive=False,
                                                                           show_copy_button=True)
                    with gr.Accordion(label="評価結果", visible=False,
                                      open=True) as tab_chat_document_llm_openai_gpt4_evaluation_accordion:
                        tab_chat_document_openai_gpt4_evaluation_text = gr.Textbox(label="評価結果", show_label=False,
                                                                                   lines=2,
                                                                                   autoscroll=True, interactive=False,
                                                                                   show_copy_button=True)
                with gr.Accordion(label="Claude 3 Opus メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_document_llm_claude_3_opus_accordion:
                    tab_chat_document_claude_3_opus_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                             lines=2,
                                                                             autoscroll=True, interactive=False,
                                                                             show_copy_button=True)
                    with gr.Accordion(label="評価結果", visible=False,
                                      open=True) as tab_chat_document_llm_claude_3_opus_evaluation_accordion:
                        tab_chat_document_claude_3_opus_evaluation_text = gr.Textbox(label="評価結果", show_label=False,
                                                                                     lines=2,
                                                                                     autoscroll=True, interactive=False,
                                                                                     show_copy_button=True)
                with gr.Accordion(label="Claude 3.5 Sonnet メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_document_llm_claude_3_sonnet_accordion:
                    tab_chat_document_claude_3_sonnet_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                               lines=2,
                                                                               autoscroll=True, interactive=False,
                                                                               show_copy_button=True)
                    with gr.Accordion(label="評価結果", visible=False,
                                      open=True) as tab_chat_document_llm_claude_3_sonnet_evaluation_accordion:
                        tab_chat_document_claude_3_sonnet_evaluation_text = gr.Textbox(label="評価結果",
                                                                                       show_label=False,
                                                                                       lines=2,
                                                                                       autoscroll=True,
                                                                                       interactive=False,
                                                                                       show_copy_button=True)
                with gr.Accordion(label="Claude 3 Haiku メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_document_llm_claude_3_haiku_accordion:
                    tab_chat_document_claude_3_haiku_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                              lines=2,
                                                                              autoscroll=True, interactive=False,
                                                                              show_copy_button=True)
                    with gr.Accordion(label="評価結果", visible=False,
                                      open=True) as tab_chat_document_llm_claude_3_haiku_evaluation_accordion:
                        tab_chat_document_claude_3_haiku_evaluation_text = gr.Textbox(label="評価結果",
                                                                                      show_label=False,
                                                                                      lines=2,
                                                                                      autoscroll=True,
                                                                                      interactive=False,
                                                                                      show_copy_button=True)

    gr.Markdown(value="### Developed by Oracle Japan", elem_classes="sub_Header")
    tab_create_oci_clear_button.add([tab_create_oci_cred_user_ocid_text, tab_create_oci_cred_tenancy_ocid_text,
                                     tab_create_oci_cred_fingerprint_text,
                                     tab_create_oci_cred_private_key_file])
    tab_create_oci_cred_button.click(create_oci_cred,
                                     [tab_create_oci_cred_user_ocid_text, tab_create_oci_cred_tenancy_ocid_text,
                                      tab_create_oci_cred_fingerprint_text,
                                      tab_create_oci_cred_private_key_file],
                                     [tab_create_oci_cred_sql_accordion, tab_create_oci_cred_sql_text])
    tab_create_oci_cred_test_button.click(test_oci_cred, [tab_create_oci_cred_test_query_text],
                                          [tab_create_oci_cred_test_vector_text])
    tab_create_cohere_cred_button.click(create_cohere_cred,
                                        [tab_create_cohere_cred_api_key_text],
                                        [tab_create_cohere_cred_api_key_text])
    tab_create_openai_cred_button.click(create_openai_cred,
                                        [tab_create_openai_cred_base_url_text, tab_create_openai_cred_api_key_text],
                                        [tab_create_openai_cred_base_url_text, tab_create_openai_cred_api_key_text])
    tab_create_claude_cred_button.click(create_claude_cred,
                                        [tab_create_claude_cred_api_key_text],
                                        [tab_create_claude_cred_api_key_text])
    tab_create_langfuse_cred_button.click(create_langfuse_cred,
                                          [tab_create_langfuse_cred_secret_key_text,
                                           tab_create_langfuse_cred_public_key_text,
                                           tab_create_langfuse_cred_host_text],
                                          [tab_create_langfuse_cred_secret_key_text,
                                           tab_create_langfuse_cred_public_key_text,
                                           tab_create_langfuse_cred_host_text])
    tab_chat_with_llm_answer_checkbox_group.change(set_chat_llm_answer, [tab_chat_with_llm_answer_checkbox_group],
                                                   [tab_chat_with_llm_command_r_accordion,
                                                    tab_chat_with_llm_command_r_plus_accordion,
                                                    tab_chat_with_llm_openai_gpt4o_accordion,
                                                    tab_chat_with_llm_openai_gpt4_accordion,
                                                    tab_chat_with_llm_claude_3_opus_accordion,
                                                    tab_chat_with_llm_claude_3_sonnet_accordion,
                                                    tab_chat_with_llm_claude_3_haiku_accordion])
    tab_chat_with_llm_clear_button.add(
        [tab_chat_with_llm_query_text, tab_chat_with_llm_answer_checkbox_group,
         tab_chat_with_command_r_answer_text, tab_chat_with_command_r_plus_answer_text,
         tab_chat_with_openai_gpt4o_answer_text, tab_chat_with_openai_gpt4_answer_text,
         tab_chat_with_claude_3_opus_answer_text, tab_chat_with_claude_3_sonnet_answer_text,
         tab_chat_with_claude_3_haiku_answer_text
         ])
    tab_chat_with_llm_chat_button.click(chat_stream,
                                        inputs=[tab_chat_with_llm_system_text,
                                                tab_chat_with_llm_query_text,
                                                tab_chat_with_llm_answer_checkbox_group],
                                        outputs=[tab_chat_with_command_r_answer_text,
                                                 tab_chat_with_command_r_plus_answer_text,
                                                 tab_chat_with_openai_gpt4o_answer_text,
                                                 tab_chat_with_openai_gpt4_answer_text,
                                                 tab_chat_with_claude_3_opus_answer_text,
                                                 tab_chat_with_claude_3_sonnet_answer_text,
                                                 tab_chat_with_claude_3_haiku_answer_text
                                                 ])
    tab_create_table_button.click(create_table, [], [tab_create_table_sql_accordion, tab_create_table_sql_text])
    tab_load_document_load_button.click(load_document,
                                        inputs=[tab_load_document_file_text, tab_load_document_server_directory_text],
                                        outputs=[tab_load_document_output_sql_text, tab_load_document_doc_id_text,
                                                 tab_load_document_page_count_text,
                                                 tab_load_document_page_content_text],
                                        )
    tab_split_document.select(refresh_doc_list,
                              outputs=[tab_split_document_doc_id_radio, tab_delete_document_doc_ids_checkbox_group,
                                       tab_chat_document_doc_id_checkbox_group])
    tab_split_document_split_button.click(split_document_by_unstructured,
                                          inputs=[tab_split_document_doc_id_radio, tab_split_document_chunks_by_radio,
                                                  tab_split_document_chunks_max_slider,
                                                  tab_split_document_chunks_overlap_slider,
                                                  tab_split_document_chunks_split_by_radio,
                                                  tab_split_document_chunks_split_by_custom_text,
                                                  tab_split_document_chunks_language_radio,
                                                  tab_split_document_chunks_normalize_radio,
                                                  tab_split_document_chunks_normalize_options_checkbox_group],
                                          outputs=[tab_split_document_output_sql_text, tab_split_document_chunks_count,
                                                   tab_split_document_chunks_result_dataframe],
                                          )
    tab_split_document_embed_save_button.click(split_document_by_unstructured,
                                               inputs=[tab_split_document_doc_id_radio,
                                                       tab_split_document_chunks_by_radio,
                                                       tab_split_document_chunks_max_slider,
                                                       tab_split_document_chunks_overlap_slider,
                                                       tab_split_document_chunks_split_by_radio,
                                                       tab_split_document_chunks_split_by_custom_text,
                                                       tab_split_document_chunks_language_radio,
                                                       tab_split_document_chunks_normalize_radio,
                                                       tab_split_document_chunks_normalize_options_checkbox_group],
                                               outputs=[tab_split_document_output_sql_text,
                                                        tab_split_document_chunks_count,
                                                        tab_split_document_chunks_result_dataframe],
                                               ).then(embed_save_document_by_unstructured,
                                                      inputs=[tab_split_document_doc_id_radio,
                                                              tab_split_document_chunks_by_radio,
                                                              tab_split_document_chunks_max_slider,
                                                              tab_split_document_chunks_overlap_slider,
                                                              tab_split_document_chunks_split_by_radio,
                                                              tab_split_document_chunks_split_by_custom_text,
                                                              tab_split_document_chunks_language_radio,
                                                              tab_split_document_chunks_normalize_radio,
                                                              tab_split_document_chunks_normalize_options_checkbox_group],
                                                      outputs=[tab_split_document_output_sql_text,
                                                               tab_split_document_chunks_count,
                                                               tab_split_document_chunks_result_dataframe],
                                                      )
    tab_delete_document.select(refresh_doc_list,
                               outputs=[tab_split_document_doc_id_radio, tab_delete_document_doc_ids_checkbox_group,
                                        tab_chat_document_doc_id_checkbox_group])
    tab_delete_document_delete_button.click(delete_document,
                                            inputs=[tab_delete_document_server_directory_text,
                                                    tab_delete_document_doc_ids_checkbox_group],
                                            outputs=[tab_delete_document_delete_sql,
                                                     tab_split_document_doc_id_radio,
                                                     tab_delete_document_doc_ids_checkbox_group])
    tab_chat_document.select(refresh_doc_list,
                             outputs=[tab_split_document_doc_id_radio, tab_delete_document_doc_ids_checkbox_group,
                                      tab_chat_document_doc_id_checkbox_group])
    tab_chat_document_doc_id_all_checkbox.change(
        lambda x: gr.CheckboxGroup(interactive=False, value="") if x else gr.CheckboxGroup(
            interactive=True, value=""),
        tab_chat_document_doc_id_all_checkbox, tab_chat_document_doc_id_checkbox_group)
    tab_chat_document_llm_answer_checkbox_group.change(set_chat_llm_answer,
                                                       [tab_chat_document_llm_answer_checkbox_group],
                                                       [tab_chat_document_llm_command_r_accordion,
                                                        tab_chat_document_llm_command_r_plus_accordion,
                                                        tab_chat_document_llm_openai_gpt4o_accordion,
                                                        tab_chat_document_llm_openai_gpt4_accordion,
                                                        tab_chat_document_llm_claude_3_opus_accordion,
                                                        tab_chat_document_llm_claude_3_sonnet_accordion,
                                                        tab_chat_document_llm_claude_3_haiku_accordion])
    tab_chat_document_llm_evaluation_checkbox.change(
        lambda x: (
            gr.Textbox(visible=True, interactive=True),
            gr.Textbox(visible=True, value="", interactive=True)) if x else (
            gr.Textbox(visible=False, interactive=False), gr.Textbox(visible=False,
                                                                     interactive=False, value="")),
        tab_chat_document_llm_evaluation_checkbox,
        [tab_chat_document_system_message_text, tab_chat_document_standard_answer_text]
    ).then(set_chat_llm_evaluation,
           [
               tab_chat_document_llm_evaluation_checkbox],
           [
               tab_chat_document_llm_command_r_evaluation_accordion,
               tab_chat_document_llm_command_r_plus_evaluation_accordion,
               tab_chat_document_llm_openai_gpt4o_evaluation_accordion,
               tab_chat_document_llm_openai_gpt4_evaluation_accordion,
               tab_chat_document_llm_claude_3_opus_evaluation_accordion,
               tab_chat_document_llm_claude_3_sonnet_evaluation_accordion,
               tab_chat_document_llm_claude_3_haiku_evaluation_accordion])
    tab_chat_document_generate_query_radio.change(
        lambda x: (gr.Textbox(value=""), gr.Textbox(value=""), gr.Textbox(value="")),
        [tab_chat_document_generate_query_radio],
        [tab_chat_document_sub_query1_text, tab_chat_document_sub_query2_text,
         tab_chat_document_sub_query3_text])
    tab_chat_document_chat_document_button.click(generate_query, [tab_chat_document_query_text,
                                                                  tab_chat_document_generate_query_radio],
                                                 [tab_chat_document_sub_query1_text,
                                                  tab_chat_document_sub_query2_text,
                                                  tab_chat_document_sub_query3_text]
                                                 ).then(search_document,
                                                        [tab_chat_document_reranker_model_radio,
                                                         tab_chat_document_reranker_top_k_slider,
                                                         tab_chat_document_reranker_threshold_slider,
                                                         tab_chat_document_threshold_value_slider,
                                                         tab_chat_document_top_k_slider,
                                                         tab_chat_document_doc_id_all_checkbox,
                                                         tab_chat_document_doc_id_checkbox_group,
                                                         tab_chat_document_text_search_checkbox,
                                                         tab_chat_document_text_search_k_slider,
                                                         tab_chat_document_query_text,
                                                         tab_chat_document_sub_query1_text,
                                                         tab_chat_document_sub_query2_text,
                                                         tab_chat_document_sub_query3_text,
                                                         tab_chat_document_partition_by_k_slider,
                                                         tab_chat_document_answer_by_one_checkbox,
                                                         tab_chat_document_extend_first_chunk_size,
                                                         tab_chat_document_extend_around_chunk_size],
                                                        [tab_chat_document_output_sql_text,
                                                         tab_chat_document_searched_data_summary_text,
                                                         tab_chat_document_searched_result_dataframe]
                                                        ).then(chat_document,
                                                               inputs=[
                                                                   tab_chat_document_searched_result_dataframe,
                                                                   tab_chat_document_llm_answer_checkbox_group,
                                                                   tab_chat_document_query_text],
                                                               outputs=[tab_chat_document_command_r_answer_text,
                                                                        tab_chat_document_command_r_plus_answer_text,
                                                                        tab_chat_document_openai_gpt4o_answer_text,
                                                                        tab_chat_document_openai_gpt4_answer_text,
                                                                        tab_chat_document_claude_3_opus_answer_text,
                                                                        tab_chat_document_claude_3_sonnet_answer_text,
                                                                        tab_chat_document_claude_3_haiku_answer_text]
                                                               ).then(eval_ragas,
                                                                      inputs=[
                                                                          tab_chat_document_llm_answer_checkbox_group,
                                                                          tab_chat_document_llm_evaluation_checkbox,
                                                                          tab_chat_document_system_message_text,
                                                                          tab_chat_document_standard_answer_text,
                                                                          tab_chat_document_command_r_answer_text,
                                                                          tab_chat_document_command_r_plus_answer_text,
                                                                          tab_chat_document_openai_gpt4o_answer_text,
                                                                          tab_chat_document_openai_gpt4_answer_text,
                                                                          tab_chat_document_claude_3_opus_answer_text,
                                                                          tab_chat_document_claude_3_sonnet_answer_text,
                                                                          tab_chat_document_claude_3_haiku_answer_text],
                                                                      outputs=[
                                                                          tab_chat_document_command_r_evaluation_text,
                                                                          tab_chat_document_command_r_plus_evaluation_text,
                                                                          tab_chat_document_openai_gpt4o_evaluation_text,
                                                                          tab_chat_document_openai_gpt4_evaluation_text,
                                                                          tab_chat_document_claude_3_opus_evaluation_text,
                                                                          tab_chat_document_claude_3_sonnet_evaluation_text,
                                                                          tab_chat_document_claude_3_haiku_evaluation_text]
                                                                      ).then(generate_download_file,
                                                                             [
                                                                                 tab_chat_document_searched_result_dataframe,
                                                                                 tab_chat_document_llm_answer_checkbox_group,
                                                                                 tab_chat_document_llm_evaluation_checkbox,
                                                                                 tab_chat_document_query_text,
                                                                                 tab_chat_document_standard_answer_text,
                                                                                 tab_chat_document_command_r_answer_text,
                                                                                 tab_chat_document_command_r_plus_answer_text,
                                                                                 tab_chat_document_openai_gpt4o_answer_text,
                                                                                 tab_chat_document_openai_gpt4_answer_text,
                                                                                 tab_chat_document_claude_3_opus_answer_text,
                                                                                 tab_chat_document_claude_3_sonnet_answer_text,
                                                                                 tab_chat_document_claude_3_haiku_answer_text,
                                                                                 tab_chat_document_command_r_evaluation_text,
                                                                                 tab_chat_document_command_r_plus_evaluation_text,
                                                                                 tab_chat_document_openai_gpt4o_evaluation_text,
                                                                                 tab_chat_document_openai_gpt4_evaluation_text,
                                                                                 tab_chat_document_claude_3_opus_evaluation_text,
                                                                                 tab_chat_document_claude_3_sonnet_evaluation_text,
                                                                                 tab_chat_document_claude_3_haiku_evaluation_text
                                                                             ],
                                                                             [tab_chat_document_download_output_button])

app.queue()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        max_threads=200,
        show_api=False,
        auth=do_auth,
    )
