import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from Tips import *
import json
import threading
#############################################
LLMProcessor=None
LLMModel=None
#############################################加载LLM模型
def LoadLLM():
    global LLMModel,LLMProcessor
    dir="./Safetensors/deepseek"
    LLMProcessor = AutoTokenizer.from_pretrained(dir)
    LLMModel = AutoModelForCausalLM.from_pretrained(
        dir,
        device_map=0, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval()
######################################API调用
def AnswerTextByAPI(role_tip:str,question:str):
    client=client1()
    completion = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model="qwen-flash",
        messages=[
            {"role":"system","content":f"{role_tip}"},
            {"role": "user", "content": f"{question}"},
        ],
    )
    return completion.choices[0].message.content
#########################################本地调用
def AnswerTextByPipe(role_tip:str,question:str):
    ############加载模型
    LoadLLM()
    # 4. 定义对话内容
    messages=[
            {"role":"system","content":f"{role_tip}"},
            {"role": "user", "content": f"{question}"},
    ],
    global LLMModel,LLMProcessor
    # 5. 处理输入（转为模型可识别的格式，并移到模型所在设备）
    inputs = LLMProcessor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(DEVICE)  # 确保输入与模型在同一设备（GPU）

    # 6. 生成回复（添加推理参数优化效果）
    outputs = LLMModel.generate(
        **inputs,
        max_new_tokens=3000,  # 最大生成 token 数
        temperature=0.7,  # 随机性（0-1，越低越确定）
        do_sample=True,  # 启用采样生成（更自然）
        pad_token_id=LLMProcessor.eos_token_id  # 填充 token 指定
    )

    # 7. 解码并打印回复（只取新增生成的部分）
    response = LLMProcessor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],  # 跳过输入部分
        skip_special_tokens=True  # 忽略特殊 token（如 <bos> <eos>）
    )
    #
    idx0,idx1=response.find("<think>"),response.find("</think>")
    think=response[idx0+7:idx1]
    answer=response[idx1+8:]
    Debug("AnswerTextByPipe深度思考:",think)
    #卸载模型
    del outputs
    del inputs
    del LLMModel 
    del LLMProcessor
    gc.collect()
    torch.cuda.empty_cache()
    #
    return answer
#########################################调用
def AnswerText(role_tip:str,question:str):
    try:
        if Enable_Local_LLM:
            return AnswerTextByPipe(role_tip,question)
        else:
            return AnswerTextByAPI(role_tip,question)
    except Exception as e:
        Debug("AnswerText:",e)
        return AnswerText(role_tip,question)

if __name__=="__main__":
    while True:
        try:
            prompt=input("prompt:")
            res=AnswerText("",prompt)
            print(res)
        except Exception as e:
            print(e)