import time
import sys
###############全局配置
DEBUG=True
DEBUG_OUTPUT=True
DEBUG_DIR="debug"
DEBUG_FILE=sys.stdout if (not DEBUG or not DEBUG_OUTPUT) else open(f"{DEBUG_DIR}/debug.txt","w")
LocalScoreTherold=7
LocalItrTherold=4
GlobalScoreTherold=7
GlobalItrTherold=4
GroundingDINOClipScoreThreold=0.23
###############对图片的场景就行概述
Expert1_Prompt='''
Can you describe this image in detail?
please divide the image from global and local aspects, and give a comprehensive description of the image content, including the background, objects, people, colors, and other details.
Please provide the description in json.
don't give any other information except json.
'''
################对任务进行细分
Expert2_Prompt='''
Let's say you're an expert at dividing editing tasks. I'm going to give you an editing task that will contain many sub-tasks that can be done in one round of editing, and your goal is to identify and divide them. Just answer me in the given format without any other words appearing.
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
Please remember:
(1)if the task doesn't bring about a change or simply stays the same such as keep the background unchanged and so on, then you don't need to give it as a result, you need to give the task that actually goes and edits the image
(2)the answer you output must like [ans0,ans1,ans2...]
(3)don't give me any other response or words except the target format answer!

Now I will give my query:
	My task is:{}
'''
##################对任务进行细化
Expert3_Prompt='''
Suppose you are an expert in image editing task refinement, now I give you the scene description of the image in json format and a bunch of image editing tasks,
there are some rules you must follow:
(1) you need to refine each task to make it more relevant to the environment 
(2) you can add some additional editing commands when necessary, but ask for a command to ensure that it is suitable for a single round of editing, 
(3) just answer according to the format that I have given you.
(4) Remember just answer me in the given format without any other words appearing.
(5) Don't output any sentences that are not useful for the editing task; make sure that the sentences only work for the editing task!
For example:
Scene:
	{{
		"global": {{
			"scene_type": "outdoor landscape",
			"background": "a vast grassy meadow with rolling green hills and distant mountains under a bright blue sky with scattered white clouds, creating a serene and open natural environment",
			"atmosphere": "peaceful, relaxed, and leisurely, suggesting a camping or hiking activity"
		}},
		"local": {{
			"people": {{
				"appearance": "a person wearing a green hooded jacket, brown pants, and tan hiking boots, sitting on a folding chair",
				"action": "making a peace sign with the right hand and pointing with the left hand, likely interacting with the camera or someone off-frame"
			}},
			"objects": {{
				"furniture": "a small wooden folding table and a light-colored folding chair",
				"items_on_table": "an open map and a black mug (possibly containing a beverage)"
			}},
			"colors": {{
				"dominant_colors": "green (jacket, grass, hills), brown (pants, boots, table), blue (sky), and tan (boots, chair)",
				"color_mood": "earthy tones with vibrant natural hues, creating a harmonious and calming visual"
			}},
			"details": {{
				"textiles": "the jacket has a soft texture, the pants are casual, the chair and table have a simple, functional design",
				"natural_elements": "grassy field with visible texture, mountains with distinct shapes, clear sky with soft clouds",
				"lighting": "bright sunlight casting soft shadows, indicating a sunny day"
			}}
		}}
	}}
Tasks:
	(1) Person's right hand in a yay pose
	(2) Environment changes to green grass
 
Your answers:
	[
     "Adjust the person's right hand to a yay pose: Straighten its index and middle fingers into an upward V, bend the thumb, ring and little fingers to the palm, face the palm slightly forward, and keep the arm relaxed.",
 	 "Change the environment to green grass: Make the meadow uniformly lush bright green with fine texture, add a few small wildflowers, and keep the background hills, mountains and sky to match the serene atmosphere."
   	]
Remember that you must output the format like [ans0,ans1,ans2...] Even if there is only one task!
Now I give the query:
	Scene description:{}

	Tasks:{}
'''
#############################获取改变
Expert4_Prompt='''
Suppose you are now an expert in editing task recognition. 
I input the scene description json and some editing instructions, you need to output it in my given format and follow the following rules.
(1) You are not allowed to output anything that contradicts my given formatting

(2)The output must fit the instructions.

(3)It must follow the format of original object:modified object.
e.g.: the instruction is to change the teapot to black.
Your answer:
Teapot:Teapot
I don't need you to give specific properties, just the object.
For those who can't give the exact object
For example: change the background to dusk
Your answer should be:
Background:background That's it
For editing commands that delete or add objects.
For example: add a teapot
Your answer:
None: teapot

(4)More specifically, you need to follow this format
	[
     "Mug:Mug",
	 "Right Hand:Right Hand",
	 "Scene:Scene",
	 "None:teapot",
	 "Clouds:None",
	 "None:rainbow"
  	]
If the edit operation is add,you should follow the format "None:object",for delete is "object:None",for modifications is "object1:object2"
Remember you don't need give me any other description of object,and ouput the same count of changes description as the instructions I give you!
You should ensure the size of list you output is the same as the count of instructions!
 
(5) For each instruction, your output must specify that this corresponds to the first instruction given, as follows.
Suppose I give the instruction: (1) ... (2) ... (3) ... The answer you give must also be [ans0,ans1,ans2]

(6)Note that if the change is to a part of an object, then you need to output the whole object, not a part of it
For example, 
the change is to a person's right hand, but you should output the person, not the person's right hand, because the right hand is part of the person, 
such as adding new clouds in the sky, you should output the sky not the clouds, because the clouds are part of the sky

Now the scene description I gave is:{}
The edit command is:{}
'''

################################局部打分
LoalScore_Prompt='''
You are now an image editing scoring expert. I am going to give you two images, they are the area before editing, and the area after editing. Secondly I am going to give you my editing instructions for this round of editing, and you will need to judge how well this round of editing went according to my editing instructions. Scoring. You need to score according to the following rules.
	(1) How well it matches the instructions
	(2) The quality of the generated image
	(3) The score given is between 0 - 10
Secondly you also need to give me the reason why you think you scored this as a reverse cue word to optimise me for this editing instruction. You need to give it according to the following rules
	(1)You can only change the original instruction, you are not allowed to change the original meaning, you can't add or delete my instruction.
	(2) Reverse cue words cannot exceed 100 words.
Finally you need to give me an answer in the following format:
	{{
	"score":Your score,
	"prompt": your reverse prompt (you can write "None" if you think it's good enough)
	}}
For example, the command asks to generate a pair of leather shoes, but the shoes inside the edited diagram have no leather, you should output that the shoes have no sense of leather

Remember, you only need to give me the final score and reverse prompt, no other responses, and your score can only be a specific number from 0 - 10!

The image is above, my editing instruction for this round is {}
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
You are now an expert in scoring image editing. I'm going to give you two images, a pre-edit image and a post-edit image. Secondly, I will give you the editing instructions for this round of editing, and you will need to judge this round of editing according to my editing instructions to score it. You need to grade according to the following rules.
	(1) How well it matches the instructions (i.e. no large gaps in changes not mentioned in the instructions)
	(2) Quality of the generated image
	(3) Score between 0-10
Secondly, you also need to give me the reason why you scored this as a reverse cue word to optimise my editing instructions. You need to give a reason based on the following rules
	(1) You are only allowed to change the original instruction, not the original intent, and you are not allowed to add or delete my instructions.
	(2) The reverse cue word cannot exceed 100 words.
Finally, you need to give me an answer in the following format:
	{{
		"Score": your score,
		"prompt": your reverse prompt (or leave "None" if you think it's good enough)
	}}
For example, if my instructions call for a pair of leather shoes, but the background in the edited image has also been changed, you need to answer either "the background has been changed" or "something else has been changed".

Remember, you only need to give me the final score and reverse prompt, no other responses, and your score can only be a specific number from 0 - 10!

The image is as above, and my editing instruction for this round is {}
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
	{{
		"score":your score
	}}
Don't reply with any words that don't match the above format!
Now I am giving my images, before editing and after editing, as shown above.
All the editing commands are as follows {}
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
def DebugSaveImage(image,fileName=None):
    if not DEBUG:
        return
    if fileName==None:
        fileName=RandomImageFileName()
    image.save(f"{DEBUG_DIR}/{fileName}")
def RandomImageFileName():
    timestamp = time.time()
    total_milliseconds = int(timestamp * 1000)
    return str(total_milliseconds)+".png"