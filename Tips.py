import time
import sys
from torch import cuda
from volcenginesdkarkruntime import Ark
import os
from openai import OpenAI
from functools import partial
from io import BytesIO
import base64
from PIL import Image
import time
import threading
from PIL import Image,ImageFilter,ImageDraw
###############全局配置
os.system("rm -rf debug/")
os.system("mkdir debug")
DEVICE = "cuda" if cuda.is_available() else "cpu"
TEST_MODE=True	#测试模式将验证测试机
PARALLE_MODE=TEST_MODE and True  #并行测试所有的数据集
THREAD_OBJECT=None if not PARALLE_MODE else threading.local() #存储线程级别的对象数据
TEST_CNT=20
DEBUG=True
DEBUG_LOCK=None if not DEBUG else threading.Lock()
DEBUG_OUTPUT=True
DEBUG_DIR="debug/"
DEBUG_FILE=sys.stdout if (not DEBUG or not DEBUG_OUTPUT or PARALLE_MODE) else open(f"{DEBUG_DIR}/debug.txt","w",encoding="utf-8")
GlobalScoreThershold=7
GlobalItrThershold=3
ClipScoreThreshold=0.21
Enable_Local_LLM=False
Enable_Local_VLM=False
Enable_Local_ImageEdit=False
Enable_TaskPolish=False
Enable_TextureFix=False
#################the type of task
TaskType=[
        "add",
        "remove",
        "replace",
        "global_style_transfer",
        "perspective_shift",
        "attribute_change",
        "move",
        "modify",
        "background_change"
        ]
TaskTypeExpress=[
    "Introduceanewobject,person,orelementintotheimage,e.g.:addacarontheroad",
    "Eliminateanexistingobjectorelementfromtheimage,e.g.: removethesofaintheimage",
    "Substitute one object in the image with a different object, e.g.: replace the coffee with an apple",
    "Modify the entire image to adopt a different visual style, e.g.: make the style of the image to cartoon",
    "Change the perspective of the image,e.g.: Overlooking the whole playground",
    "Change object's attribute such as it's color、size and texture,e.g.: Change ball's color to red and make it bigger",
    "Change the spatial position of an object within the image, e.g.: move the plane to the left",
    "Change object's action 、style、expression、apperance",
    "Change the background of the image,e.g.: Change the background to a forest"
]
TaskType=[f"{TaskType[i]} : ( {TaskTypeExpress[i]} )" for i in range(len(TaskType))]
################对任务进行细分
Expert1_Prompt=f'''
Let's say you're a professional and detailed image editing task subdivider, specializing in breaking down a single comprehensive image editing task (which contains multiple interrelated yet independently executable sub-tasks that can all be completed in one round of editing) into clear, specific, and actionable individual sub-editing instructions. Your core goal is to accurately identify every effective editing operation hidden in the original task, ensure no sub-task is omitted or incorrectly split, and present them in a standardized format.

### Key Operating Rules (Must Be Strictly Followed):
1. **Definition of Valid Sub-Tasks**: Only include sub-tasks that truly modify the image (i.e., bring about changes to elements in the image). Any content that requires keeping elements unchanged, maintaining the original state, or not performing edits (such as "keep the background color unchanged", "do not adjust the size of the chair", "maintain the original style of the picture frame") must be completely excluded and not listed in the result.
2. **Standard for Sub-Task Division**: Each sub-task must target **one specific element** in the image (e.g., a teacup, a person's right hand, the environment, a teapot, clouds in the sky, a car, a person's hairstyle, a table's material, window curtains, etc.) and execute **one single editing operation** (e.g., changing color, adjusting posture, replacing the environment, adding an object, deleting an object, replacing an object, modifying material, adjusting brightness, changing hairstyle, etc.). Sub-tasks must be independent of each other—completing one sub-task does not rely on another, and each can be executed separately.
3. **Clarity and Specificity Requirements**: Each sub-task string must be concise, clear, and free of ambiguous expressions. Avoid vague descriptions (e.g., do not use "fix the cup" but instead "change the color of the cup to red"; do not use "adjust the person" but instead "make the person's left hand hold a book"). Ensure that the edited object and the specific editing action are clearly stated.
4. **Example Reference**: For instance, if the original task is "change the color of the sofa to gray, let the child hold a teddy bear, remove the potted plant on the table, and replace the wall painting with a landscape photo", the subdivided sub-tasks should be:
   - "Change the color of the sofa to gray."
   - "Make the child hold a teddy bear."
   - "Remove the potted plant on the table."
   - "Replace the wall painting with a landscape photo."
5. **You need to output the type of editing instructions after fine segmentation, including one of the following operations:
    {TaskType}.
### Format Requirements (Non-Negotiable):
- You must only output the subdivided sub-tasks in an array [] format. Each sub-task is a separate string enclosed in double quotes, and commas are used to separate different sub-task strings.
- Do not add any additional content outside the array (such as explanations, prompts, notes, or greetings). Even if the original task has only one sub-task, it must still be placed in the array.
- The array format must be consistent with the example (neatly formatted, with each sub-task string on a new line for readability, but ensuring the syntax of the array is correct).

Example:
Question: change the colour of the teacup to black and have the person's right hand in a yay pose, change the setting to a green meadow, add a teapot, delete the clouds in the sky and replace them with rainbows
Your answers:
[
 ["The colour of the teacup is changed to black.","modify"],
 ["The person's right hand makes a yay pose.","modify"],
 ["The environment is changed to a green meadow","modify"],
 ["Add a teapot","add"],
 ["delete the clouds in the sky","remove"],
 ["add a rainbow in the sky","add"]
]
'''
###################################利用反向提示词优化指令
InsOptim_Prompt='''
You are now an expert in optimising image editing instructions. I give you postive prompt and image editing instructions, you need to optimise my editing instructions based on the postive prompt and output new edit prompt according to the following rules.
	(1) You cannot change the original meaning of my instructions
	(2) You can't add or delete on my instructions
	(3)You can only modify the part linked to the postive promp
	(4) Your output should follow this format
		{
			"new_instruction":your modified instruction
		}
For example:
    postive prompt:Makes man's shoes less wrinkled
    edit prompt:change man's shoes into  leather shoes
    output:{
        "new_instruction":"change man's shoes into  leather shoes with little wrinkled"
    }
    
Remember that you can't output any words that don't match this format!
'''
###################################全局打分
GlobalScore_Prompt='''
You are now an expert in scoring image editing. I'm going to give you two images, a pre-edit image and a post-edit image. 
Secondly, I will give you the editing instructions for this round of editing, and you will need to judge this round of editing according to my editing instructions to score it. 
Your task:
    You need to grade according to the following rules.
        (1) How well it matches the instructions (i.e. no large gaps in changes not mentioned in the instructions)
        (2) Quality of the generated image
        (3) Score between 0-10
    You need to give me "negative prompt" and "positive prompt"  in edited image according to the following rules..
        (1) The prompt  cannot exceed 100 words,The simpler the better.
        (2) The negative prompt is what you don't want in image,so if you don't want a dog,you should output "dog" instead of "not draw a dog".
        (3) The negative prompt can be directly used for image-edit model as negative prompt.
        (4) For negative prompt,you need to tell where it is wrong instead such as "red shirt" or "thick beef" 
        (5) You shouldn't output "not" or "don't" or any other negative word in  negative prompt because negative prompt is something went wrong which don't match my expection.
    You need to give me an answer in the following format:
	{
		"score": your score,
		"negative_prompt": your negative prompt (or leave "None" if you think it's good enough),
		"positive_prompt":your positive prompt
	}
For example:
Example_1:
    tasks:remove the dog
    issue:the background also changed into grass land
    Output:
    {
        "score":5,
        "negative_prompt":"background changed" ,
        "positive_prompt": "remove the dog clearly while keep other unchanged"
    }
Example_2:
    tasks:add clouds in sky
    issue:a sun also generate
    Output:
    {
        "score":3,
        "negative_prompt":"sun",
        "positive_prompt": "add clouds in sky while keep other unchanged"
    }
Example_3:
    tasks:add some flowers in background,
    issue:The flowers added are chrysanthemums, and I want ornamental flowers.
    Output:
    {
        "score":0,
        "negative_prompt":"chrysanthemums",
        "positive_prompt": "add some flowers in background in particular ornamental flower"
    }
Example_4:
    tasks:move man's hands over his head
    issue:man's hands hang down
    Output:
    {
        "score":0,
        "negative_prompt":"man's hands hang down",
        "positive_prompt":"move man's hands higher over his head "
    }
Example_5:
    tasks:add some steaks to the grill
    issue:The original steak texture has been altered and the added steaks' texture is much red than original
    Output:
    {
        "score":0,
        "negative_prompt":"much red steak",
        "positive_prompt":"add some steaks to the grill while keep other steak's shape in good appearance"
    }
Remember, you only need to give me the final score and negative prompt, no other responses, and your score can only be a specific number from 0 - 10!
Remember,You don't need to give me any explanations in any other place such as after prompt or score
'''
######################################评估器
Critic_Prompt='''
You are now an expert in image editing evaluation. I will now give you two images, a pre-edited image and a post-edited image, and will also enter all the commands I have used for this one multi-round edit, and you will need to synthesise the editing commands, compare the two images, and finally give me a rating. The scoring criteria are as follows.
	(1) how well it matches the instructions (i.e., there can be no changes other than those in the instructions, but also make sure that the instructions were executed well)
	(2) Quality of the image, resolution, clarity, etc.
	(3) the overall aesthetics of the generated images
	(4) The reasonableness of the content of the generated images
Your score must be within 1-10.
Your answer should be in the following format
	{
		"score":your score
	}
Don't reply with any words that don't match the above format!
'''
################################反馈总结提示
PromptFeedbackSummary_Prompt='''
You are now an expert image editor and are good at summarising the feedback I give on several image edits. I'm going to give you an array of input feedback and then you need to summarise that feedback according to the following rules.
1. Where the feedback is mentioned, the summary you give must be fully inclusive
2. You are not allowed to add anything that is not mentioned in the other feedbacks or delete anything that is mentioned in the feedbacks.
3. The answer you give should be short and concise at the core.
4. You must follow the following json format output
    {
        "prompt":your answer
    }
Example.
    [don't make shoes too large,shoes's colour must be red]
Your answer.
    {
        "prompt": "shoes's must be red and not too large"
    }
Remember to answer only in the above format and do not give any sentence other than the specified answer!
'''
################################指令优化
EDIT_SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the edited input image’s overall scene.  

## 2. Task Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Do not translate or alter the original language of the text, and do not change the capitalization.  
- **For text replacement tasks, always use the fixed template:**
    - `Replace "xx" to "yy"`.  
    - `Replace the xx bounding box to "yy"`.  
- If the user does not specify text content, infer and add concise text based on the instruction and the input image’s context. For example:  
    > Original: "Add a line of text" (poster)  
    > Rewritten: "Add text \"LIMITED EDITION\" at the top center with slight shadow"  
- Specify text position, color, and layout in a concise way.  

### 3. Human Editing Tasks
- Maintain the person’s core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.).  
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.  
- **For expression changes, they must be natural and subtle, never exaggerated.**  
- If deletion is not specifically emphasized, the most important subject in the original image (e.g., a person, an animal) should be preserved.
    - For background change tasks, emphasize maintaining subject consistency at first.  
- Example:  
    > Original: "Change the person’s hat"  
    > Rewritten: "Replace the man’s hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged"  

### 4. Style Transformation or Enhancement Tasks
- If a style is specified, describe it concisely with key visual traits. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco: flashing lights, disco ball, mirrored walls, colorful tones"  
- If the instruction says "use reference style" or "keep current style," analyze the input image, extract main features (color, composition, texture, lighting, art style), and integrate them into the prompt.  
- **For coloring tasks, including restoring old photos, always use the fixed template:** "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration"  
- If there are other changes, place the style description at the end.

## 3. Rationality and Logic Checks
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" should be logically corrected.  
- Add missing key information: if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edges).  

# Output Format Example
```json
{
   "Rewritten": "..."
}
'''
################################给inpainting生成区域
GetMaskArea_Prompt='''
You are now an image object bounding box detection expert. I will provide you with an image and a prompt of the image-edit instruction, and you need to give the answer in accordance with the following rules:
    (1) If there are multiple target objects or area, return multiple results; if there is only one, return one result;.
    (2) For each result, provide a four-tuple (x0, y0, x1, y1), where each element is a floating-point number between 0 and 1, representing the relative position of the target from its top-left corner to bottom-right corner in the image.
    (3) The final result should follow the following format: [(ans0), (ans1), ...]
    (4) Each answer may represent an area as a mask for inpainting or an area bounding the target object.
    (5) Separate each element with a comma
Example:
    task:add some steaks on grill like others
    your answer:[(0.1,0.2,0.3,0.4),(0.2,0.2,0.3,0.3),(0.1,0.1,0.2,0.2)]  (These areas is empty and good enough to add one steak on it)
Attention:
    (1)You should ensure the area you give as mask for inpainting will work good for the instruction
    (2)If the instruction is the operation of add object ,the area must match the realistic dimensions of the object.
Example:
    task:add a t-shirt on man's left hand
    your answer:[(0.1,0.2,0.3,0.5)],it's big enough for inpainting model to generate a t-shirt there.
Remember: Only need to provide the answer, without any additional responses.
    '''
#################################获取编辑任务里面的操作对象
ObjectGet_Prompt='''
    You are now an expert in object extraction for image editing commands, and you need to extract the object I need to operate from it based on the editing commands I enter, following the following rules.
    (1) The description of the operation object given must be as detailed as possible.
    (2) If there is more than one object, you need to return more than one result.
    (3) The answer you give shouldn't include the operation of the object.
    (4) You must give the answer in the following format
        Format:[ans0, ans1, ans2...]
    Example: [ans0,ans1,ans2...].
        Example_1.
            Task: Remove the bird with red feathers.
            Your answer: ["bird with red feathers"].
        Example_2.
            Task:Remove the ball from the table and change the colour of the green broom to red.
            Your answer: ["ball on the table", "green broom"].
    Remember, don't answer anything other than what I've specified!
'''
################################预定义的反面提示词
PreDefine_NegPrompt=''''''
'''
Worst quality, Normal quality, Low quality, Low res, Blurry, Jpeg artifacts, Grainy, text, logo, 
watermark, banner, extra digits, signature, subtitling, Bad anatomy, Bad proportions, Deformed, 
Disconnected limbs, Disfigured, Extra arms, Extra limbs, Extra hands, Fused fingers, Gross proportions, 
Long neck, Malformed limbs, Mutated, Mutated hands, Mutated limbs, Missing arms, Missing fingers, Poorly drawn hands, 
Poorly drawn face, Nsfw, Uncensored, Cleavage, Nude, Nipples, Overexposed, Plain background, Grainy, Underexposed, Deformed structures
'''
################################纹理修复提示词
TextureFix_Prompt='''
You are now a master of image editing texture repair, and I'm going to give you the original image, the edited image, my image editing instructions, and the negative feedback that you need to determine which negative feedback can be eliminated by texture repair as a way of repairing the image texture.
Specifically, you need to follow the following rules.
(1) The set of textures you give that need to be repaired can only come from my negative feedback set.
(2) You must give your answer in the following format
    [area0,area1...]
Example:
    I give you instruction and negative feedback as.
        instruction:remove the mobile phone in man's hand
        negative feedback:["The face was distorted", "The colour of the vase was changed", "The position of the hand was changed"].
    Your answer.
    ["face", "vase"]
You may not give any other answer than the required format answer.
'''
################################调试函数
def Debug(*msg):
    if not DEBUG:
        return
    #
    try:
        DEBUG_LOCK.acquire()
        original=sys.stdout
        sys.stdout=THREAD_OBJECT.logfile if PARALLE_MODE else DEBUG_FILE
        for x in msg:
            print(x,end='',flush=True)
        print()
        sys.stdout=original
    finally:
        DEBUG_LOCK.release()
        
def DebugSaveImage(image,fileName=None,dir=DEBUG_DIR):
    if not DEBUG:
        return
    if fileName==None:
        fileName=RandomImageFileName()
    image.save(f"{dir}/{fileName}")
    
def RandomImageFileName():
    timestamp = time.time()
    total_milliseconds = int(timestamp * 1000)
    return str(total_milliseconds)+".png"

def client0():
    client=Ark(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key="a63a43cf-5056-4cae-ad94-11e4a82e7447",
        timeout=1800,
        # 设置重试次数
        max_retries=2,
    )
    return client
def client1():
    client= OpenAI(
        api_key="sk-17cd5f2ebd6b4981b9eb6991a0ddfe3d",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=1800,
        max_retries=2,
    )
    return client
#编码图片
def encode_image(pil_image:Image.Image)->str:
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:{'image/jpeg'};base64,{encoded_string}"
#性能分析
class Timer():
    def __init__(self):
         self.beg=time.time()
    def __call__(self, *args, **kwds):
         interval=time.time()-self.beg
         return str(int(interval))+"秒"
     
def DrawRedBox(image, boxes, width=3):
    # 拷贝原图避免修改原图像
    image_copy = image.copy()
    # 创建可绘制对象
    draw = ImageDraw.Draw(image_copy)
    # 画红框
    for box in boxes:
        draw.rectangle(box, outline="red", width=width)
    return image_copy