#!/usr/bin/env python3

import dspy

lm = dspy.HFClientVLLM(model="microsoft/Phi-3-medium-128k-instruct", port=38242, url="http://localhost", max_tokens=200)
dspy.settings.configure(lm=lm, trace=[], temperature=1.5)

def build_complete_prompt(json_object_strings, current_prompt):
    complete_prompt = ''
    for json_object_string in json_object_strings:
        complete_prompt += json_object_string
    complete_prompt += current_prompt
    return complete_prompt

task_description = "Generate a banger tweet, look at previous tweets and generate a tweet that is appropriate to the feedback and the score given."

json_object_strings = []
while True:
    prompt_reasoning = f'''
{{
    "feedback": """Great tweet! This will get a lot of engagement.""",
    "score": """5""",
    "task": """{task_description}""",
    "reasoning": """'''

    complete_prompt = build_complete_prompt(json_object_strings, prompt_reasoning)
    completion_reasoning = lm(complete_prompt, stop='"""')[0]
    prompt_tweet = f'''
{{
    "feedback": """Great tweet! This will get a lot of engagement.""",
    "score": """5""",
    "task": """{task_description}""",
    "reasoning": """{completion_reasoning}"""
    "tweet": """'''
    complete_prompt = build_complete_prompt(json_object_strings, prompt_tweet)
    completion_tweet = lm(complete_prompt, stop='"""')[0]
    print("complete_prompt:", complete_prompt)
    print('='*50)
    print("completion_reasoning:", completion_reasoning)
    print("completion_tweet:", completion_tweet)
    feedback_text = input('Write your feedback: ')
    feedback_score = input('Rate the completion from 1 to 5: ')
    complete_json_object = f'''
{{
    "task": """{task_description}""",
    "reasoning": """{completion_reasoning}""",
    "tweet": """{completion_tweet}""",
    "feedback": """{feedback_text}""",
    "score": """{feedback_score}""",
}}'''
    json_object_strings.append(complete_json_object)



