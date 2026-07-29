COLOR_PROMPT = """
You are an AI assistant who will help me to evaluate a model reasoning based on a reference Ground Truth (GT). You must grade the reasoning focusing specifically on the COLOR ACCURACY axis.

  To mark a response, output a single integer between 1 and 5. 5 means the color descriptions in the reasoning perfectly match the GT. 1 means they are completely different or contradict the GT.
  
  NEUTRALITY RULE: If the Ground Truth does not mention any specific colors, do not penalize the reasoning for including them. In such cases, assign a score of 5 as long as the color details provided do not contradict any other environmental facts mentioned in the Ground Truth.

  Example 1 (Mark: 5):
  Ground Truth: 'The room is well-lit, featuring a large mahogany desk with a polished finish. On the desk, there is a deep burgundy ceramic vase placed next to some silver-colored metallic pens.'
  Reasoning: 'I can see a dark office setup. There is a burgundy vase on the desk, which stands out against the silver stationery items nearby.'
  Your Mark: 5

  Example 2 (Mark: 3):
  Ground Truth: 'The kitchen area is modern. A bright neon yellow storage bin is tucked under the white counter, providing a sharp contrast to the grey tiled floor.'
  Reasoning: 'In the kitchen, I notice a light-colored bin placed near the floor area.'
  Your Mark: 3 (The reasoning is too vague about the specific neon yellow color mentioned in the GT).

  Example 3 (Mark: 1):
  Ground Truth: 'The living room has a navy blue rug with white patterns. To its left, a small emerald green ottoman is placed near the window.'
  Reasoning: 'The floor is covered by a large orange rug, and there is a bright red chair in the corner.'
  Your Mark: 1 (Color descriptions completely contradict the GT).

  Your Turn:
  Ground Truth (GT): {GT}
  Reasoning: {REASONING}

  Provide just the number between 1 and 5 as a response. Do not include any explanations or additional text. 
"""

TEXTURE_PROMPT = """
You are an AI assistant who will help me to evaluate a model reasoning based on a reference Ground Truth (GT). You must grade the reasoning focusing specifically on the TEXTURE AND MATERIAL axis.

  To mark a response, output a single integer between 1 and 5. 5 means the materials and textures match the GT perfectly. 1 means they are entirely incorrect.
  Provide the output in the following JSON format:
  
  NEUTRALITY RULE: If the Ground Truth is silent about materials or textures, the reasoning should not be penalized for mentioning these properties. Assign a score of 5 if the described materials are plausible and do not conflict with the objects or context described in the Ground Truth.

  Example 1 (Mark: 5):
  Ground Truth: 'The entryway features a sturdy desk made of white painted wood. The surface shows a visible grain texture. The wall behind it is composed of horizontal wooden shiplap planks.'
  Reasoning: 'The furniture appears to be a white desk with a wooden grain texture, positioned against a wall made of horizontal wood panels.'
  Your Mark: 5

  Example 2 (Mark: 3):
  Ground Truth: 'The bathroom counter is made of smooth, cold grey marble with light white veining. The faucet is made of brushed stainless steel.'
  Reasoning: 'The counter is a grey stone surface, and the sink fixtures look like they are made of metal.'
  Your Mark: 3 (Correct general materials, but misses the specific marble texture and steel type).

  Example 3 (Mark: 1):
  Ground Truth: 'The patio has a rough concrete floor and a heavy wrought-iron table in the center.'
  Reasoning: 'The scene shows a soft carpeted area with a lightweight plastic folding table.'
  Your Mark: 1 (Completely different materials and textures).

  Your Turn:
  Ground Truth (GT): {GT}
  Reasoning: {REASONING}

  Provide just the number between 1 and 5 as a response. Do not include any explanations or additional text. 
"""

SPATIAL_PROMPT = """
You are an AI assistant who will help me to evaluate a model reasoning based on a reference Ground Truth (GT). You must grade the reasoning focusing specifically on the CONTEXT AND SPATIAL FEATURES axis.

  To mark a response, output a single integer between 1 and 5. 5 means the contextual relations match the GT perfectly. 1 means the context is fundamentally wrong.
  Provide the output in the following JSON format:
    
  NEUTRALITY RULE: If the Ground Truth does not specify the surrounding objects or spatial relationships, do not penalize the reasoning for providing this context. Assign a score of 5 if these contextual details remain consistent with the overall scene described in the Ground Truth.

  Example 1 (Mark: 5):
  Ground Truth: 'In the workspace, a computer monitor is placed directly in front of an ergonomic chair. To the right of the monitor, there is a small desk lamp and a stack of notebooks.'
  Reasoning: 'The setup includes a monitor and a chair. I also see a lamp and some books positioned on the right side of the screen.'
  Your Mark: 5

  Example 2 (Mark: 3):
  Ground Truth: 'The kitchen features a microwave sitting on a standalone wooden cart. Above the cart, there is a set of open shelves holding various ceramic mugs.'
  Reasoning: 'I see a microwave in the kitchen near some shelves and some dishes.'
  Your Mark: 3 (Misses the specific spatial relationship of the 'standalone cart' and 'above' placement).

  Example 3 (Mark: 1):
  Ground Truth: 'There is a large potted plant sitting by the window in the far left corner of the room, away from the television set.'
  Reasoning: 'The plant is placed directly on top of the television stand in the center of the wall.'
  Your Mark: 1 (The spatial context and object placement are opposite to the GT).

  Your Turn:
  Ground Truth (GT): {GT}
  Reasoning: {REASONING}

  Provide just the number between 1 and 5 as a response. Do not include any explanations or additional text. 
"""


INSTRUCTION = ("You are a robot navigating an enclosed space."
" Your goal is to navigate to the correct object based on the user's commands. You were given the following task by the"
" user '{TASK}'. Currently, you are facing a scene represented by the given image. Reason about what you are seeing,"
" comparing what you know about the task (given the user commands) and the given scene. For example, if the task is"
" 'Navigate to the black leather sofa near a lampstand' your reasoning process will be"
" 'I'm currently observing a brown sofa which is different than"
" black, making it unlikely to be the target sofa. Moreover, there"
" is no lampstand near it, only a rug and a window' etc. If there"
" are distortions or artifact, do not focus on them, focus on the"
" object at hand. At the end of the reasoning process, evaluate"
" how well the provided image aligns with the user's task. Assign"
" a confidence score based on the following scale: - 0: You are"
" certain the image DOES NOT match the task. - 1: You are unsure"
" whether the image matches the task or not. - 2: You are certain"
" the image DOES match the task. Provide a concise reasoning"
" (under 100 words) and strictly follow this output format:\n"
"<motivation>Your reasoning here</motivation><score>0, 1, or 2</score>")
