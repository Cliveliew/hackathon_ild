import os
import gradio as gr
import pandas as pd
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import base64
import re

def extra_code(text) -> dict:
    # 使用正则表达式提取代码块
    code_snippet = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)

    if code_snippet:
        # 提取到的代码片段
        extracted_code = code_snippet.group(0)
    else:
        extracted_code = "没有找到Python代码块"
    return extracted_code

def extra_json(text) -> dict:
    # 使用正则表达式提取json
    code_snippet = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)

    if code_snippet:
        # 提取到的json片段
        extracted_json = code_snippet.group(0)
        extracted_json = extracted_json.replace("```json\n", "")
        extracted_json = extracted_json.replace("\n```", "")
    else:
        extracted_json = "没有找到json块"
    return extracted_json

import sys
from io import StringIO
import contextlib

import re

def execute_code(code: str) -> dict:
    code = code.replace("```python\n", "")
    code_content = code.replace("\n```", "")
    # 创建一个StringIO对象来捕获输出
    output = StringIO()

    try:
        # 使用contextlib捕获标准输出
        with contextlib.redirect_stdout(output):
            # 执行代码
            exec(code_content)

        # 获取输出结果
        execution_output = output.getvalue()
    except Exception as e:
        execution_output = "Code running error:" + str(e)
    
    return str(execution_output)


# 设置 NVIDIA AI Endpoints 的 API 密钥
os.environ["NVIDIA_API_KEY"] = "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"

# 初始化 ChatNVIDIA 模型
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", streaming=True)  # 启用流式输出
llm2 = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", streaming=True)  # 启用流式输出

# 创建带历史记录的对话提示模板
prompt = ChatPromptTemplate.from_template(
"""
你是一位专门从事Excel的数据分析专家，精通数据处理和分析。你的主要工作是根据以下用户历史意图和用户的最新输入，结合表头信息，合理确定用户的当前需求：

##用户最新输入、表头信息及历史意图:
{input}

##输出：
```json
{{
    "current_request": "",  // 用户的最新需求
    "request_type": []  // 用户需求的类型，为'统计'、'绘图'、'修改'中的一个或若干个，其中修改表示用户需求中包括增删改的操作。
}}
```

##分析步骤：
1.首先判断当前查询是否是完整的独立需求：
-是否包含明确的分析对象和分析动作
-是否能独立理解和执行
-是否是一个新的独立问题或统计需求

2.如果当前查询不完整，再判断与历史查询的关联性：
-是否使用代词(这个、那个、它等)指代上文内容
-是否是对上一个问题的补充/延伸
-是否使用省略句依赖上下文才能理解
-是否是对原有需求的替换而非叠加


输出规则：
1. 如果当前查询是完整的独立需求：
   直接输出当前查询内容，不需要与历史需求叠加

2. 如果当前查询不完整且与上次查询有关联：
   - 若是补充/延伸：结合上下文补全完整意图
   - 若是替换：仅保留被替换部分的上下文

注意事项：
1. 图表需求必须在当前查询中明确提出，不自动继承历史图表需求
2. 对于"最大值"、"最小值"、"总数"、"平均值"等统计类查询，通常作为独立需求处理
3. 避免过度叠加历史需求，确保输出的意图简洁且符合用户当前真实需求
4.直接输出用户意图即可，无需输出分析过程

示例：
示例1: 
用户历史意图: "起运港维度统计出运次数"
用户输入: "出运次数最多？"
输出: // 上下文相关，新输入为旧需求的补充
```json
{{
    "current_request": "找出出运次数最多的起运港",
    "request_type": ["统计"]
}}
```

示例2:
用户历史意图: "起运港维度统计出运次数，并找出出运次数最多的起运港"
用户输入: "起运港总数？"
输出: // 上下文相关，但新输入为新需求
```json
{{
    "current_request": "统计起运港总数",
    "request_type": ["统计"]
}}
```

示例3:
用户历史意图: "按船司统计箱量"
用户输入: "生成饼图"
输出: // 上下文相关，新输入为旧需求的补充
```json
{{
    "current_request": "按船司统计箱量并生成饼图",
    "request_type": ["统计", "绘图"]
}}
```

示例4:
用户历史意图: "按船司统计箱量并生成饼图"
用户输入: "绘制柱状图"
输出: // 上下文相关，新输入为旧需求的替换
```json
{{
    "current_request": "按船司统计箱量并生成柱状图",
    "request_type": ["统计", "绘图"]
}}
```

注意：请勿给出分析过程，直接输出结果json即可。
"""
)

# 创建不带历史记录的对话提示模板
prompt2 = ChatPromptTemplate.from_template(
    """
    你是一位专门从事Excel的数据分析专家，精通数据处理和分析。你的主要工作是根据用户提供的Excel数据，提供表格数据的总结概述。
    请注意，输出结果时，无需输出计算过程，只需输出结论即可。
    请注意，不要给出具体的数值，描述数据分布即可。

    表格信息:
    {input}
    """
)

# 初始化对话内存 - 只为第一个chain配置
# memory = ConversationBufferMemory()  # 这将自动处理 history

# 创建带历史记录的对话链
conversation_chain = prompt | llm | StrOutputParser()
# ConversationChain(
#     llm=llm,
#     prompt=prompt,
#     memory=memory,
#     output_parser=StrOutputParser(),
# )

# 创建不带历史记录的对话链
conversation_chain2 = prompt2 | llm2 | StrOutputParser()


# -------------------------------输出统计代码的llm--------------------------------
llm_calculate = ChatNVIDIA(model="qwen/qwen2.5-coder-32b-instruct", streaming=True)  # 启用流式输出

# 生成统计代码的对话提示模板
prompt_calculate = ChatPromptTemplate.from_template(
"""
你是一位专门从事Excel的数据分析专家，精通数据处理和分析。你的主要工作是基于用户提供的excel表头信息，分析用户需求，编写基于pandas的python代码。
{input}

请注意，你的编码习惯为尽量避免构造函数，而是使用通俗易懂的，用户可以理解的代码进行编写。
请注意，若用户输入中包含绘制图表的需求，你无需响应，后续有绘制图表的专有流程，你只需专注于数据统计和数据展示。你只需要使用pandas库，请勿调用matplotlib等绘图库。
请注意，若用户输入中包含修改表格的需求，你无需响应，后续有修改的专有流程，你只需专注于数据统计和数据展示。
##你的处理步骤如下：
1.pandas读取用户输入的excel文件
2.根据用户意图，拆解步骤，并编写python脚本代码
##示例
默认的excel文件路径：./1.xlsx
表头信息：['作业号', '出运类型', '操作员']
现在，请你以作业号和出运类型的组合，统计出运次数
输出示例：
'''
import pandas as pd 

# 读取Excel文件 
file_path = './1.xlsx' 
df = pd.read_excel(file_path)

# 按“作业号”和“出运类型”组合，统计出运次数 
out_count = df.groupby(['作业号', '出运类型']).size().reset_index(name='出运次数') 

# 生成markdown表格
markdown = "### 出运次数统计\\n\\n"
# 添加表头
headers = "|" + "|".join(str(col) for col in out_count.columns) + "|"
separator = "|" + "|".join(["---" for _ in out_count.columns]) + "|"

markdown += headers + "\\n" + separator + "\\n"

# 添加数据行
for _, row in out_count.iterrows():
    markdown += "|" + "|".join(str(val) for val in row.values) + "|\\n"

# 输出markdown格式的结果
print(markdown)
'''

注意：仅需生成代码即可，无需输出分析过程和其他无关信息。

"""
)

# 创建生成统计代码的对话链
conversation_chain_calculate = prompt_calculate | llm_calculate | StrOutputParser()
# ---------------------------------------------------------------------------------------------

# -------------------------------输出绘图代码的llm--------------------------------
llm_draw = ChatNVIDIA(model="qwen/qwen2.5-coder-32b-instruct", streaming=True)  # 启用流式输出

# 生成统计代码的对话提示模板
prompt_draw = ChatPromptTemplate.from_template(
"""
你是一位专门从事Excel的数据分析专家，精通数据处理和分析，尤其擅长各类图表的绘制。你的主要工作是基于用户提供的excel表头信息，分析用户绘图需求，编写基于pandas、matplotlib的python图形绘制代码。
请注意，你的编码习惯为尽量避免构造函数，而是使用通俗易懂的，用户可以理解的代码进行编写。
用户输入及表头信息：
{input}

##你的处理步骤如下：
1.pandas读取用户输入的excel文件
2.根据用户意图，拆解步骤，并编写数据分析的python代码
3.根据数据分析的结果，基于matplotlib库，绘制各类图表
4.你应首先确立图表的x、y轴，确定好维度，拟定合适的图表名称，保存图表图片
5.打印图片的文件路径
##示例
默认的excel文件路径：./1.xlsx
表头信息：['作业号', '出运类型', '操作员', '销售员', '委托部门', '船名', '航次', 'ETD日期', '船东', '起运港名称', '卸货港名称', '目的地名称', '箱型', 'TEU']
用户需求：现在，请你按起运港与卸货港组合，统计出运次数，并绘制饼状图
输出示例：
'''python
import pandas as pd
import matplotlib.pyplot as plt
import tempfile

# 1. 读取 Excel 文件
# 默认文件名为 './1.xlsx'，且数据在第一个 Sheet 中
file_path = './1.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 2. 数据处理：按起运港与卸货港组合统计出运次数
# 创建一个新列 '起运港-卸货港'，用于组合起运港和卸货港
df['起运港-卸货港'] = df['起运港名称'] + ' - ' + df['卸货港名称']

# 统计每个组合的出运次数
port_combination_counts = df['起运港-卸货港'].value_counts()

# 设置字体路径，避免中文乱码
from matplotlib.font_manager import FontProperties
font_path = r'C:\Windows\Fonts\simhei.ttf'  
font = FontProperties(fname=font_path, size=12)

# 3. 绘制饼图
plt.figure(figsize=(10, 8))  # 设置图表大小
plt.pie(
    port_combination_counts, 
    labels=port_combination_counts.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    textprops={{'fontproperties': font}}  # 通过 textprops 设置字体
)
plt.title('起运港与卸货港组合的出运次数占比', fontproperties=font)  # 设置图表标题
plt.axis('equal')  # 使饼图为正圆形

img_name = ""
# 将图表保存为临时文件
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
    plt.savefig(tmpfile.name, format="png", bbox_inches='tight')  # 保存图表
    plt.close()
    img_name = tmpfile.name
print(img_name)
'''

注意：仅需生成代码即可，无需输出分析过程和其他无关信息。
"""
)

# 创建生成统计代码的对话链
conversation_chain_draw = prompt_draw | llm_draw | StrOutputParser()
# ---------------------------------------------------------------------------------------------

# 定义读取 Excel 文件的函数
def preview_excel(file_path):
    try:
        # 读取 Excel 文件
        df = pd.read_excel(file_path)

        # 获取前5行数据作为样本
        sample_df = df.head()

        # 生成 Markdown 表格
        markdown = "\n\n### 数据样本预览\n\n"
        markdown += sample_df.to_markdown(index=False)

        # 添加数据集信息
        markdown += f"\n### 数据集信息\n"
        markdown += f"- 总行数: {len(df)}\n"
        markdown += f"- 总列数: {len(df.columns)}\n"
        markdown += f"- 列名: {', '.join(df.columns.tolist())}\n"
        column_name = df.columns.tolist()
        return markdown, df, column_name

    except Exception as e:
        return f"错误: {str(e)}", None

# 定义分析 Excel 的函数（支持流式输出）
def analyze_excel(data_input):
    # 使用不带历史记录的对话链
    for chunk in conversation_chain2.stream({"input": data_input}):
        yield chunk

# 定义文件上传后的处理逻辑（支持流式输出）
def handle_file(file, history):
    if file is None:
        return history, history, None, None, None
        
    # 读取 Excel 文件并生成预览
    markdown, df, column_name = preview_excel(file.name)

    # 调用大模型分析表格内容并总结（流式输出）
    response_first_calling = ""
    for chunk in analyze_excel(str(df)):
        response_first_calling += chunk
        updated_history = history + [("数据分析总结", response_first_calling)]
        yield updated_history, updated_history, df, column_name, file.name
    
    # # 将数据预览添加到聊天记录中
    # history.append(("上传文件", markdown))
    response_first_calling += markdown
    updated_history = history + [("数据分析总结", response_first_calling)]
    yield updated_history, updated_history, df, column_name, file.name
    # yield history, history, df
    

# 将图片文件转换为 base64 编码
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# 定义流式输出的函数
def stream_response(input_text, history=None, column_name=None, df=None, idea_list=None, file_name=None):
    if history is None:
        history = []
    
    # 将历史记录转换为字符串格式
    # history_str = "\n".join([f"用户: {user}\n机器人: {bot}" for user, bot in history])
    human_history = [user.replace("Human: ", "") for user, bot in history]
    # history_str = [f"{user.replace("Human: ", "")}" for user, bot in history]
    history_str = str(human_history)
    
    # # 如果用户上传了文件，先读取文件
    # if df is not None:
    #     # 将数据集的列名和基本信息添加到提示中
    #     input_text += f"\n当前数据集列名: {', '.join(df.columns.tolist())}"
    
    # 调用对话链生成流式回复
    response = ""

    tmp_column_name = str(column_name)
    if len(idea_list) > 0:
        tmp_idea = idea_list[-1]
        tmp_idea_list = "用户的意图为：{}， 用户的意图类型为：{}".format(tmp_idea['current_request'], tmp_idea['request_type'])
    else:
        tmp_idea_list = ""
    tmp_input_text = """
    用户输入：{}，
    表头信息：{}，
    用户历史意图：{}
    """.format(input_text, tmp_column_name, tmp_idea_list)
    
    # print(tmp_input_text)

    response += "##用户意图为：\n"
    # print(history_str)
    for chunk in conversation_chain.stream({"input": tmp_input_text}):
        response += chunk
        yield history + [(input_text,  response)], history + [(input_text, response)], df, column_name, idea_list

    try:
        format_response = extra_json(response)
        # print(format_response)
        format_response = eval(format_response)
        # stream_response = "用户的意图为：{}， 用户的意图类型为：{}".format(format_response['current_request'], format_response['request_type'])
        idea_list.append(format_response)

        if len(idea_list) > 0:
            custom_request = idea_list[-1]
            crt_request_type = custom_request['request_type']
            crt_current_request = custom_request['current_request']

            if "统计" in crt_request_type:
                message = "用户需求：{} \n表头信息：{}\n".format(crt_current_request, column_name)
                
                response += "\n##生成的统计代码为：\n"
                tmp_code_response = ""
                for chunk_code in conversation_chain_calculate.stream({"input": message}):
                    response += chunk_code
                    tmp_code_response += chunk_code
                    yield history + [(input_text, response)], history + [(input_text,response)], df, column_name, idea_list

                output_code = extra_code(tmp_code_response)
                # print("file_name", file_name)
                output_code = output_code.replace("'./1.xlsx'", "r'{}'".format(file_name))
                # print("------------\n {} ------------------".format(output_code))
                execute_code_res = execute_code(output_code)
                # print("------------\n {} ------------------".format(execute_code_res))
                response += "\n##统计结果\n{}\n".format(execute_code_res)

                yield history + [(input_text, response)], history + [(input_text,response)], df, column_name, idea_list
                # for updated_history, _, _, column_name, idea_list in stream_response_calcute_code(message, chat_history, column_name, df, idea_list):
                #     yield "", updated_history, df, column_name, idea_list


            if "绘图" in crt_request_type:
                message = "用户需求：{} \n表头信息：{}\n".format(crt_current_request, column_name)
                
                response += "\n##生成的绘图代码为：\n"
                tmp_code_response = ""
                for chunk_code in conversation_chain_draw.stream({"input": message}):
                    response += chunk_code
                    tmp_code_response += chunk_code
                    yield history + [(input_text, response)], history + [(input_text,response)], df, column_name, idea_list

                output_code = extra_code(tmp_code_response)
                # print("file_name", file_name)
                output_code = output_code.replace("'./1.xlsx'", "r'{}'".format(file_name))
                # print("------------\n {} ------------------".format(output_code))
                execute_code_res = execute_code(output_code)
                # print("------------\n {} ------------------".format(execute_code_res))
                # execute_code_res = execute_code_res.replace("\\", "/")
                # print("图片路径：", execute_code_res)
                    # 将图片转换为 base64 编码
                execute_code_res = execute_code_res[:-1]
                image_base64 = image_to_base64(execute_code_res)

                draw_res_markdown = f"![图表](data:image/png;base64,{image_base64})"
                
                response += "\n##绘图结果\n{}\n".format(draw_res_markdown)

                yield history + [(input_text, response)], history + [(input_text,response)], df, column_name, idea_list

    except Exception as e:
        print(e)
        


# 定义统计代码生成llm的流式输出的函数
def stream_response_calcute_code(input_text, history=None, column_name=None, df=None, idea_list=None):
    # 如果用户上传了文件，先读取文件
    # if df is not None:
    #     # 将数据集的列名和基本信息添加到提示中
    #     input_text += f"\n当前数据集列名: {', '.join(df.columns.tolist())}"
    
    # 调用对话链生成流式回复
    response = ""
    for chunk in conversation_chain_calculate.stream({"input": input_text}):
        response += chunk
        yield history + [(input_text, response)], history + [(input_text, response)], df



# 创建 Gradio 界面
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 数据分析聊天机器人")
        
        # 文件上传组件
        file_input = gr.File(label="上传 Excel 文件", file_types=[".xlsx", ".xls"])
        
        # 聊天记录显示组件
        chatbot = gr.Chatbot(label="对话记录")
        
        # 用户输入组件
        msg = gr.Textbox(label="输入消息")
        
        # 清除历史按钮
        clear = gr.Button("清除历史")
        
        # 数据集状态（用于存储当前加载的数据）
        df_state = gr.State()
        
        # 定义文件上传后的处理逻辑
        def handle_file_wrapper(file, history):
            for updated_history, _, _, column_name, filename in handle_file(file, history):
                yield updated_history, updated_history, df_state.value, column_name, filename
        
        # 定义消息处理逻辑
        def respond(message, chat_history, column_name, df, idea_list, file_name):
            # 使用流式输出
            for updated_history, _, _, column_name, idea_list in stream_response(message, chat_history, column_name, df, idea_list, file_name):
                yield "", updated_history, df, column_name, idea_list

            # for i in updated_history:
            #     print(i)
            
        column_name = gr.State(value=[])
        idea_list = gr.State(value=[])
        filename = gr.State(value="")
        # 绑定文件上传和消息输入
        file_input.upload(handle_file_wrapper, [file_input, chatbot], [chatbot, chatbot, df_state, column_name, filename])
        msg.submit(respond, [msg, chatbot, column_name, df_state, idea_list, filename], [msg, chatbot, df_state, column_name, idea_list])
        
        # 清除历史记录
        clear.click(lambda: ([], None), None, [chatbot, df_state], queue=False)
    
    return demo

# 启动 Gradio 应用
if __name__ == "__main__":
    interface = gradio_interface()
    # interface.launch(debug=True, share=False, show_api=False, server_port=5000, server_name="0.0.0.0")
    interface.launch(debug=True, share=True, show_api=True, server_port=5000)
