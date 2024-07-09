import os
import json
import ast
import re

def llama3_process(result):
    if result[:2]=="[]":
        print("here")
        return "[]"
    elif result[:5]=="Error":
        print("error")
        return "[]"
    # 使用正则表达式找到第一个字典
    # print("==========")
    # print(result)
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    return dict_str

def glm4_process(result):
    if result[:8]=="The task" or result[:12]=="This problem" or result[:11]=="The problem":
        print("here")
        return "[]"
    print("==========")
    print(result)
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    return dict_str

def qwen2_process(result):
    if result=="":
        print("here")
        return "[]"
    print("==========")
    print(result)
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    return dict_str
def compute_f1(llm3_path,human_path):
    # TP,FP,FN,P,R,F1
    result = {
        "Machine Domain":[0,0,0],
        "Physical Device":[0,0,0],
        "Environment Entity":[0,0,0],
        "Design Domain":[0,0,0],
        "Requirements":[0,0,0],
        "Shared Phenomena":[0,0,0]
    }
    with open(llm3_path,'r',encoding='utf-8') as file:
        llm3_data = json.load(file)
    with open(human_path,'r',encoding='utf-8') as file:
        human_data = json.load(file)
    for i,predict in enumerate(llm3_data):
        predict = glm4_process(predict['predict'])
        if predict=="[]":
            predict = {
                "Machine Domain":[],
                "Physical Device":[],
                "Environment Entity":[],
                "Design Domain":[],
                "Requirements":[],
                "Shared Phenomena":[]
            }
        else:
            predict = ast.literal_eval(predict)
        ground = human_data[i]['entity']
        for key in result.keys():
            flat_predict = [item for item in predict[key]]
            flat_ground = [item for item in ground[key]]
            TP = len(set(flat_ground).intersection(set(flat_predict)))
            FP = len(set(flat_predict)) - TP
            FN = len(set(flat_ground)) - TP
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
    llm3_path = "/home/jindongming/project/modeling/PDEval/output/entity/glm4-9b/fold_0/1.json"
    human_path = "/home/jindongming/project/modeling/PDEval/data/dataset/10-fold/fold_0/test_data.json"
    compute_f1(llm3_path,human_path)

if __name__=="__main__":
    main()