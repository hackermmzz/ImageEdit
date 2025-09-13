import time
import sys
from torch import cuda
from volcenginesdkarkruntime import Ark
import os
from openai import OpenAI
###############全局配置
os.system("rm -rf debug/*")
DEVICE = "cuda" if cuda.is_available() else "cpu"
TEST_MODE=True	#测试模式将验证测试机
DEBUG=True
DEBUG_OUTPUT=True
DEBUG_DIR="debug/"
DEBUG_FILE=sys.stdout if (not DEBUG or not DEBUG_OUTPUT) else open(f"{DEBUG_DIR}/debug.txt","w")
LocalScoreThershold=7
LocalItrThershold=2
GlobalScoreThershold=7
GlobalItrThershold=2
ClipScoreThreshold=0.21
###############对图片的场景就行概述
Expert1_Prompt='''
Can you describe this image in detail?
please divide the image from global and local aspects, and give a comprehensive description of the image content, including the background, objects, people, colors, and other details.
Please provide the description in json.
don't give any other information except json.
'''
################对任务进行细分
Expert2_Prompt='''
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

### Format Requirements (Non-Negotiable):
- You must only output the subdivided sub-tasks in an array [] format. Each sub-task is a separate string enclosed in double quotes, and commas are used to separate different sub-task strings.
- Do not add any additional content outside the array (such as explanations, prompts, notes, or greetings). Even if the original task has only one sub-task, it must still be placed in the array.
- The array format must be consistent with the example (neatly formatted, with each sub-task string on a new line for readability, but ensuring the syntax of the array is correct).

Example:
Question: change the colour of the teacup to black and have the person's right hand in a yay pose, change the setting to a green meadow, add a teapot, delete the clouds in the sky and replace them with rainbows
Your answers:
[
 "The colour of the teacup is changed to black.",
 "The person's right hand makes a yay pose.",
 "The environment is changed to a green meadow",
 "Add a teapot",
 "delete the clouds in the sky",
 "add a rainbow in the sky"
]

Now I will give the image and the task,you should split my task into sub-tasks by the image:
My task is:{}
'''
#############################获取改变
Expert3_Prompt='''
Suppose you are now an expert in editing task recognition. 
I input the image and one editing instruction, you need to output it in my given format and follow the following rules.
(1) You are not allowed to output anything that contradicts my given formatting
(2)The output must fit the instruction.
(3)It must follow the format of original object:modified object.
For example:
Tasks:
	make Person's right hand in a yay pose
Your answer:
	[
    	"person with a green hooded jacket",
     	"person with a yay pose",
    ]
    You should ensure what you describe will work good for GrounDingDINO and CLIP for next step work in details.
    
(4)If the edit operation is add,you should follow the format "None:object",for delete is "object:None",for modifications is "object1:object2"
 

(5)Note that if the change is to a part of an object, then you need to output the whole object, not a part of it
For example, 
the change is to a person's right hand, but you should output the person, not the person's right hand, because the right hand is part of the person, 
such as adding new clouds in the sky, you should output the sky not the clouds, because the clouds are part of the sky

(6)When you generate an answer for the change brought about by the ith instruction, you have to make sure that the change brought about by the previous i-1 instructions is also taken into account, which means that if a previous instruction changed the colour of a person's clothes to red, even if he started out with the colour green, then you would only be able to say that his clothes are red because the change brought about by these instructions is persistent
The answer you output should be such that when I use GroundingDINO to deduct this part, it is evident that the instruction has been actually executed. For example, when changing a red ring to a green one, it is obvious that you only need to provide "green ring:red ring". Another example: when moving the knife in a person's hand closer to their neck, you should output "person:person" instead of "knife" or "neck". This is because whether the knife is close to the neck is determined based on the entire person's area.

The edit command is:{}
'''

###################################利用反向提示词优化指令
InsOptim_Prompt='''
You are now an expert in optimising image editing instructions. I give you reverse cue words and image editing instructions, you need to optimise my editing instructions based on the reverse cue words and output them according to the following rules.
	(1) You cannot change the original meaning of my instructions
	(2) You can't add or delete on my instructions
	(3)You can only modify the part linked to the reverse cue word
	(4) Your output should follow this format
		{{
			"new_instruction":your modified instruction
		}}
For example, the reverse cue is Leather shoes don't have leather, so make sure he has leather when you change my instructions.

Remember that you can't output any words that don't match this format!
Now I will give my query:.
Reverse prompt word:{}
Instructions for this round of editing:{}
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
    You need to give me "negative prompt" and "positive prompt" and "area for prompt embeds mask" in edited image according to the following rules..
        (1) The prompt  cannot exceed 100 words，The simpler the better.
        (2) The negative prompt is what you don't want in image,so if you don't want a dog,you should output "dog" instead of "not draw a dog".
        (3) The negative prompt can be directly used for image-edit model as negative prompt.
        (4) The positive prompt can improve the robustness of my commands to make it work better
        (5) The area for prompt embeds mask should be as detailed as possible so that GroundingDino+SAM can work efficiently.
    You need to give me an answer in the following format:
	{{
		"score": your score,
		"negative_prompt": your negative prompt (or leave "None" if you think it's good enough),
		"positive_prompt":your positive prompt,
		"prompt_embeds_mask":Your given drawing area (If you don't think you need to give one, then you give "None".)
	}}
For example:
Example_1:
    tasks:remove the dog
    issue:the background also changed into grass land
    Output:
    {{
        "score":5,
        "negative_prompt": background changed ,
        "positive_prompt": remove the dog clearly while keep other unchanged,
        "prompt_embeds_mask":the black dog
    }}
Example_2:
    tasks:add clouds in sky
    issue:a sun also generate
    Output:
    {{
        "score":3,
        "negative_prompt":sun,
        "positive_prompt": add clouds in sky while keep other unchanged,
        "prompt_embeds_mask":the blue sky
    }}
Example_3:
    tasks:add some flowers in background,
    issue:The flowers added are chrysanthemums, and I want ornamental flowers.
    Output:
    {{
        "score":0,
        "negative_prompt":chrysanthemums,
        "positive_prompt": add some flowers in background in particular ornamental flower,
        "prompt_embeds_mask":None,
    }}
Example_4:
    tasks:move man's hands over his head
    issue:man's hands hang down
    Output:
    {{
        "score":0,
        "negative_prompt":man's hands hang down,
        "positive_prompt":move man's hands higher over his head ,
        "prompt_embeds_mask":None
    }}
Example_5:
    tasks:add some steaks to the grill
    issue:The original steak texture has been altered
    Output:
    {{
        "score":0,
        "negative_prompt":None,
        "positive_prompt":add some steaks to the grill while keep other steak's shape in good appearance,
        "prompt_embeds_mask":the grill 
    }}
Remember, you only need to give me the final score and negative prompt, no other responses, and your score can only be a specific number from 0 - 10!

The image is as above, and my editing instruction for this round is {}
'''
################################局部打分
LoalScore_Prompt=GlobalScore_Prompt
######################################评估器
Critic_Prompt='''
You are now an expert in image editing evaluation. I will now give you two images, a pre-edited image and a post-edited image, and will also enter all the commands I have used for this one multi-round edit, and you will need to synthesise the editing commands, compare the two images, and finally give me a rating. The scoring criteria are as follows.
	(1) how well it matches the instructions (i.e., there can be no changes other than those in the instructions, but also make sure that the instructions were executed well)
	(2) Quality of the image, resolution, clarity, etc.
	(3) the overall aesthetics of the generated images
	(4) The reasonableness of the content of the generated images
Your score must be within 1-10.
Your answer should be in the following format
	{{
		"score":your score
	}}
Don't reply with any words that don't match the above format!
Now I am giving my images, before editing and after editing, as shown above.
All the editing commands are as follows {}
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
{{
   "Rewritten": "..."
}}

User Input: {}

Rewritten Prompt:
'''
################################调试函数
def Debug(*msg):
	if not DEBUG:
		return
    #
	original=sys.stdout
	sys.stdout=DEBUG_FILE
    #
	for x in msg:
		print(x,end='',flush=True)
	print()
    #
	sys.stdout=original
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

client0 = Ark(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
    api_key="723cff33-3b13-420d-ab6d-267800a27475",
    timeout=1800,
    # 设置重试次数
    max_retries=2,
)
client1 = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-17cd5f2ebd6b4981b9eb6991a0ddfe3d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=1800,
    # 设置重试次数
    max_retries=2,
)