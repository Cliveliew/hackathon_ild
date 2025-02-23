import os
import streamlit as st
from PIL import Image
from paddleocr import PaddleOCR
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd

import cv2
import numpy as np

def pil_to_cv(pil_image: Image.Image) -> np.ndarray:
    """将PIL图像转换为OpenCV图像（BGR格式）"""
    # 将PIL图像转换为RGB格式的numpy数组
    rgb_image = np.array(pil_image.convert('RGB'))
    # 将RGB图像转换为BGR格式（OpenCV格式）
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image

# 设置NVIDIA API密钥
os.environ["NVIDIA_API_KEY"] = "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"

# 配置Streamlit页面
st.set_page_config(
    page_title="信息提取",
    page_icon="📝",
    layout="wide"
)

class OCRProcessor:
    """处理图片OCR识别"""
    def __init__(self, lang='ch'):
        """初始化OCR处理器，lang表示语言，use_gpu=False表示不使用GPU，强制CPU运行"""
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """预处理图片，确保最长边为500"""
        max_size = 1000
        width, height = image.size
        if width > height:
            if width > max_size:
                ratio = max_size / width
                new_width = max_size
                new_height = int(height * ratio)
            else:
                return image
        else:
            if height > max_size:
                ratio = max_size / height
                new_height = max_size
                new_width = int(width * ratio)
            else:
                return image
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def extract_text(self, image: Image.Image) -> str:
        """从图片中提取文本"""
        image = self.preprocess_image(image)
        image = pil_to_cv(image)
        result = self.ocr.ocr(image, cls=True)
        text = ""
        for line in result[0]:
            tmp_index = line[-1][0]
            text += tmp_index + "\n"
        return text.strip()


import re
def extra_json(text) -> dict:
    # 使用正则表达式提取json
    code_snippet = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    code_snippet_1 = re.search(r'```\n(.*?)\n```', text, re.DOTALL)

    
    if code_snippet:
        # 提取到的json片段
        extracted_json = code_snippet.group(0)
        extracted_json = extracted_json.replace("```json\n", "")
        extracted_json = extracted_json.replace("\n```", "")
    elif code_snippet_1:
        # 提取到的json片段
        extracted_json = code_snippet_1.group(0)
        extracted_json = extracted_json.replace("```\n", "")
        extracted_json = extracted_json.replace("\n```", "")
    else:
        extracted_json = text
    return extracted_json

class IdentityCardInfoExtractor:
    """使用NVIDIA大模型提取信息"""
    def __init__(self, deal_type, deal_dict):
        self.llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
        name_list = deal_dict.split(" ")

        tmp_dict = str(name_list)  # 转换为字符串格式
        deal_type = str(deal_type)

        # 修正：创建包含键值对的字典
        pmt_text = "你是一个专业的{}信息提取助手。请从用户提供的{}识别结果中提取信息，并返回结果json。\n".format(deal_type, deal_type)
        pmt_text += "所需要的属性名为：{}\n".format(tmp_dict)
        pmt_text += """
输出的json示例：
'''json
{{
    "需求的属性名1": "对应的属性值",
    "需求的属性名2": "对应的属性值"
}}
'''
请严格按照上述格式输出。
请勿输出思考过程和其他无关内容，直接输出结果json即可。
    """
            
        print(pmt_text)

        # 创建ChatPromptTemplate时，直接传递字典格式
        self.prompt_template = ChatPromptTemplate.from_messages([ 
            ("system", pmt_text),
            ("user", "{input}")
        ])
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def extract_info(self, text: str) -> str:
        """从文本中提取身份证信息"""
        result = self.chain.invoke(text)
        return result


def save_to_txt(content: str, filename="信息.txt"):
    """将提取的信息保存为TXT文件"""
    try:
        st.download_button(
            label="下载TXT文件",
            data=content.strip(),
            file_name=filename,
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"保存TXT时出错: {str(e)}")


def main():
    """主函数，处理Streamlit UI逻辑"""
    st.title("信息提取系统")

    # 获取deal_type和deal_dict的输入
    deal_type = st.text_input("请输入deal_type", "身份证")
    deal_dict = st.text_input("请输入deal_dict", "身份证识别")

    uploaded_file = st.file_uploader("请上传图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption="上传的图片", use_container_width=True)

        # 初始化OCR处理器和信息提取器
        ocr_processor = OCRProcessor(lang='ch')
        # print(deal_type)
        # print(deal_dict)
        info_extractor = IdentityCardInfoExtractor(deal_type, deal_dict)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("提取信息"):
                with st.spinner("正在处理..."):
                    # 使用OCR提取文本
                    extracted_text = ocr_processor.extract_text(image)

                    if not extracted_text.strip():
                        st.error("未能提取到有效的文本，请确保图片清晰。")
                        return

                    # 使用大模型提取身份证信息
                    result = info_extractor.extract_info(extracted_text)

                    # 显示提取结果
                    with col2:
                        st.success("信息提取成功！")
                        print(result)
                        res_json = extra_json(result)
                        try:
                            st.json(res_json)
                        except:
                            st.text("输出json结果失败")
                        # 保存结果为TXT文件
                        save_to_txt(result)

if __name__ == "__main__":
    main()
