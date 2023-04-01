import pickle
import json
data = []
for i in range(100):
    try:
        temp = pickle.load(open("collected_data/quora_chat_{}.pkl".format(i),'rb'))
    except:
        continue
    for y in temp:
        x = temp[y]
        x = x.split("[Human]")[1:-1]
        if len(x)!= 0:
            s = ""
            for y in x:
                if "[AI]" in y:
                    y = y.split("[AI]")
                    if len(y) == 2:
                        s+= "[|Human|] "+ y[0].strip() + "\n" + "[|AI|] "+ y[1].strip() + "\n"
                    else:
                        break
                else:
                    break
            if s!="":
                prompt = "The conversation between human and AI assistant.\n"
                s = prompt + s + "[|Human|] "

                data.append({"input":s})

json.dump(data,open("data/quaro_chat_data.json","w"))



data = []
for i in range(100):
    try:
        temp = pickle.load(open("collected_data/stackoverflow_chat_{}.pkl".format(i),'rb'))
    except:
        continue
    for y in temp:
        x = temp[y]
        x = x.split("[Human]")[1:-1]
        if len(x)!= 0:
            s = ""
            for y in x:
                if "[AI]" in y:
                    y = y.split("[AI]")
                    if len(y) == 2:
                        s+= "[|Human|] "+ y[0].strip() + "\n" + "[|AI|] "+ y[1].strip() + "\n"
                    else:
                        break
                else:
                    break
            if s!="":
                prompt = "The conversation between human and AI assistant.\n"
                s = prompt + s + "[|Human|] "

                data.append({"input":s})
json.dump(data,open("data/stackoverflow_chat_data.json","w"))