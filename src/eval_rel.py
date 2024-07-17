import json
import os
import ast
import re

def gpt35_process(result):
    print("===========")
    print(result)
    if result=="{}":
        print("here")
        return "{}"
    else:
        return result
    pass

def mistral_process(result):
    if result=="":
        print("here")
        return "[]"
    print("==========")
    print(result)
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    print("*******")
    print(dict_str)
    return dict_str

def gpt4_process(result):
    print("===========")
    print(result)
    if result=="{}":
        print("here")
        return "{}"
    else:
        match = re.search(r"{[^}]*}", result)
        if match:
            dict_str = match.group(0)
        print("*******")
        print(dict_str)
        return dict_str


def glm4_process(result):
    print("===========")
    print(result)
    if result=="{}" or result[:4]=="Here" or result[:11]=="The problem":
        print("here")
        return "{}"
    else:
        match = re.search(r"{[^}]*}", result)
        if match:
            dict_str = match.group(0)
        print("*******")
        print(dict_str)
        return dict_str
    
def llama3_process(result):
    print("===========")
    print(result)
    if result=="{}" or result[:4]=="Here" or result[:11]=="The problem":
        print("here")
        return "{}"
    else:
        match = re.search(r"{[^}]*}", result)
        if match:
            dict_str = match.group(0)
        print("*******")
        print(dict_str)
        return dict_str
    
def qwen_process(result):
    print("===========")
    print(result)
    if result=="{}" or result[:4]=="Here" or result[:11]=="The problem":
        print("here")
        return "{}"
    else:
        match = re.search(r"{[^}]*}", result)
        if match:
            dict_str = match.group(0)
        print("*******")
        print(dict_str)
        return dict_str
    
def compute_f1(predict_path,human_path):
    result = {
        "interface":[0,0,0],
        "requirements reference":[0,0,0],
        "requirements constraints":[0,0,0]
    }
    with open(predict_path,'r',encoding='utf-8') as file:
        predict_data = json.load(file)
    with open(human_path,'r',encoding='utf-8') as file:
        human_data = json.load(file)
    for i,predict in enumerate(predict_data):
        predict = llama3_process(predict['predict'])
        if predict=="{}":
            print("I am")
            predict = {
                "interface":[],
                "requirements reference":[],
                "requirements constraints":[]
            }
        else:
            predict = ast.literal_eval(predict)
        ground = human_data[i]['relation']
        for key in result.keys():
           flat_predicted_set = {tuple(sorted(pair)) for pair in predict[key]}
           flat_ground_set = {tuple(sorted(pair)) for pair in ground[key]}
           TP = len(flat_predicted_set & flat_ground_set)
           FP = len(flat_predicted_set - flat_ground_set)
           FN = len(flat_ground_set - flat_predicted_set)
           result[key][0] += TP
           result[key][1] += FP
           result[key][2] += FN
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for key,value in result.items():
        TP = value[0]
        FP = value[1]
        FN = value[2]
        total_tp += TP
        total_fp += FP
        total_fn += FN
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall !=0 else 0
        value.append(precision)
        value.append(recall)
        value.append(f1)
    print(result)
    total_p = total_tp / (total_tp + total_fp) if total_tp + total_fp != 0 else 0
    total_r = total_tp / (total_tp + total_fn) if total_tp + total_fn !=0 else 0
    total_f1 = 2 * (total_p * total_r) / (total_p + total_r) if total_p + total_r!=0 else 0
    print(f"total_p:{total_p}, total_r:{total_r}, total_f1:{total_f1}")

def main():
    predict_path = "/home/jindongming/project/modeling/PDEval/output/relation/llama3-8b/fold_0/2.json"
    human_path = "/home/jindongming/project/modeling/PDEval/data/dataset/10-fold/fold_0/test_data.json"
    compute_f1(predict_path,human_path)

if __name__=="__main__":
    main()