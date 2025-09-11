import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from Tips import *
import json
#############################################
LLMProcessor=None
LLMModel=None
#############################################加载LLM模型
def LoadLLM():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 启用 4bit 量化
            bnb_4bit_use_double_quant=True,  # 双量化，进一步减少显存占用
            bnb_4bit_quant_type="nf4",  # 推荐的量化类型（比 fp4 更适合自然语言）
            bnb_4bit_compute_dtype=torch.bfloat16  # 计算时用 bfloat16，保精度
        )
        global LLMProcessor,LLMModel
        LLMProcessor = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
        LLMModel = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            quantization_config=bnb_config, 
            device_map="auto", 
            trust_remote_code=True
        ).to(DEVICE).eval()
#########################################调用
def AnswerText(question:str):
    ##########################调用豆包大模型
    try:
        # Non-streaming:
        completion = client.chat.completions.create(
            # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
            model="doubao-1-5-pro-256k-250115",
            messages=[
                {"role": "user", "content": f"{question}"},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        Debug(e)
        return AnswerText(question)
    ############################
        processor=LLM.processor
        model=LLM.model
        # 4. 定义对话内容
        messages = [
            {"role": "user", "content":"{}".format(question)},
        ]

        # 5. 处理输入（转为模型可识别的格式，并移到模型所在设备）
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(DEVICE)  # 确保输入与模型在同一设备（GPU）

        # 6. 生成回复（添加推理参数优化效果）
        outputs = model.generate(
            **inputs,
            max_new_tokens=65535,  # 最大生成 token 数
            temperature=0.7,  # 随机性（0-1，越低越确定）
            do_sample=True,  # 启用采样生成（更自然）
            pad_token_id=processor.eos_token_id  # 填充 token 指定
        )

        # 7. 解码并打印回复（只取新增生成的部分）
        response = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],  # 跳过输入部分
            skip_special_tokens=True  # 忽略特殊 token（如 <bos> <eos>）
        )
        return response
