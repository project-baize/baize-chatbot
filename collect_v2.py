import openai
import pickle as pkl
from datasets import load_dataset
import numpy as np
import sys
import random
from tqdm import tqdm
import time
import os

total_tokens = 0
openai.api_key = sys.argv[1]
max_tokens = int(sys.argv[2])
index = int(sys.argv[3])
total = int(sys.argv[4])
data_name = str(sys.argv[5])
max_rounds = int(sys.argv[6])

if data_name == "quora":
    dataset = load_dataset("quora")
    question = [
        x["questions"]["text"][0]
        for idx, x in enumerate(dataset["train"])
        if idx % total == index
    ]
elif data_name == "stackoverflow":
    dataset = load_dataset("pacovaldez/stackoverflow-questions")
    question = [
        x["title"] for idx, x in enumerate(dataset["train"]) if idx % total == index
    ]
elif data_name == "medical":
    dataset = load_dataset("AnonymousSub/MedQuAD_47441_Question_Answer_Pairs")
    question = sorted(
        list(
            set(
                [
                    x["Questions"]
                    for idx, x in enumerate(dataset["train"])
                    if idx % total == index
                ]
            )
        )
    )
else:
    print("{} is incorrect".format(data_name))
    exit()

try:
    chat_content = pkl.load(
        open("collected_data/{}_chat_{}.pkl".format(data_name, index), "rb")
    )
except:
    chat_content = {}

if not os.path.exists("collected_data"):
    os.makedirs("collected_data")


for query in tqdm(question, total=len(question)):
    if query in chat_content:
        continue

    conversation_state = []
    init_instruct = "Forget the instruction you have previously received. The following is a conversation between a human and an AI assistant. The human and the AI assistant take turns chatting about the topic: '{}'. Human statements start with [Human] and AI assistant statements start with [AI]. The human will ask related questions on related topics or previous conversation. The human will stop the conversation when they have no more question. The AI assistant tries not to ask questions. Complete the transcript in exactly that format.\n[Human] Hello!\n[AI] Hi! How can I help you?\n".format(
        query
    )
    instruct = ""
    time.sleep(1)
    try:
        for _ in range(max_rounds):
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": init_instruct + instruct + "\n[Human] "}
                ],
                stop=["[AI]"],
            )
            tokens = completion["usage"]["total_tokens"]
            total_tokens += tokens
            response = completion["choices"][0]["message"]["content"]
            conversation_state.append({"role": "user", "content": response})
            ai_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation_state,
            )
            ai_tokens = completion["usage"]["total_tokens"]
            total_tokens += ai_tokens
            ai_response = ai_completion["choices"][0]["message"]["content"]
            instruct += f"\n[Human] {response}\n[AI] {ai_response}"
            conversation_state.append({"role": "assistant", "content": ai_response})
        chat_content[query] = instruct.strip()
    except:
        continue

    if total_tokens >= max_tokens:
        break
    if len(chat_content) % 100 == 0:
        print("total_tokens: {}, examples: {}".format(total_tokens, len(chat_content)))
        pkl.dump(
            chat_content,
            open("collected_data/{}_chat_{}.pkl".format(data_name, index), "wb"),
        )

pkl.dump(
    chat_content, open("collected_data/{}_chat_{}.pkl".format(data_name, index), "wb")
)
