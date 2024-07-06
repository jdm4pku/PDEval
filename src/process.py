import os
import json
import spacy
import random
from sklearn.model_selection import KFold


def has_subject_verb(sentence):
    """
    func: 判断是否是完整的句子
    args: 
        sentence: 待判断的句子
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    # 遍历句子中的动词
    for token in doc:
        if token.dep_ == "ROOT":
            # 检查是否有主语
            subjects = [child for child in token.children if child.dep_ in ["nsubj", "nsubjpass"]]
            if subjects:
                return True
    return False

def process(in_dir,out_dir):
    """
    func: 对原始文本进行筛选
    args: 
        in_dir: 待筛选的文件目录(/home/jindongming/project/modeling/PDEval/data/raw_data)
        out_dir: 筛选后的文件目录(/home/jindongming/project/modeling/PDEval/data/processed_data)
    """
    for filename in os.listdir(in_dir):
        file_path = os.path.join(in_dir,filename)
        with open(file_path,'r',encoding='utf-8') as file:
            lines = file.readlines()
        print(f"before:{len(lines)}")
        lines1 = [line for line in lines if len(line)>=10]
        lines2 = [line for line in lines1 if has_subject_verb(line)]
        output_path = os.path.join(out_dir,filename)
        print(f"after:{len(lines2)}")
        with open(output_path,'w',encoding='utf-8') as file:
            for sentence in lines2:
                file.write(sentence)
            # file.write('\n'.join(lines2))

def format_human_label(in_dir,out_dir):
    """
    func: 整理人工标注的数据集的格式
    args:
        in_dir: 人工标注的数据集目录(/home/jindongming/project/modeling/PDEval/data/human_label)
        out_dir: 整理好的数据集的目录(/home/jindongming/project/modeling/PDEval/data/dataset/total)
    """
    for filename in os.listdir(in_dir):
        file_path = os.path.join(in_dir,filename)
        data = []
        with open(file_path,'r',encoding='utf-8') as file:
            lines = json.load(file)
        for line in lines:
            data_item = {}
            entity_result = {
                "Machine Domain":[],
                "Physical Device":[],
                "Environment Entity":[],
                "Design Domain":[],
                "Requirements":[],
                "Shared Phenomena":[]
            }
            relation_result = {
                "interface":[],
                "requirements reference":[],
                "requirements constraints":[]
            }
            id2entity = {}
            anno_list = line['annotations'][0]['result']
            for anno in anno_list:
                # entity
                if 'value' in anno:
                    entity = anno['value']['text'].strip()
                    label = anno['value']['labels'][0]
                    id = anno['id']
                    id2entity[id] = entity
                    entity_result[label].append(entity)
                elif 'from_id' in anno:
                    from_e = id2entity[anno['from_id']].strip()
                    to_e = id2entity[anno['to_id']].strip()
                    label = anno['labels'][0] if 'labels' in anno and len(anno['labels'])>0 else "interface"
                    relation_item = [from_e,to_e]
                    relation_result[label].append(relation_item)
            data_item['text'] = line['data']['text']
            data_item['entity'] = entity_result
            data_item['relation'] = relation_result
            data.append(data_item)
        json_data = json.dumps(data,ensure_ascii=False,indent=2)
        output_path = os.path.join(out_dir,filename)
        with open(output_path,'w',encoding='utf-8') as output_file:
            output_file.write(json_data)


# 合并不同的系统
def merge_file(in_dir,out_dir):
    """
    func: 合并不同的系统
    args:
        in_dir: "/home/jindongming/project/modeling/PDEval/data/dataset/total"
        out_dir: "/home/jindongming/project/modeling/PDEval/data/dataset/10-fold"
    """
    all_data = []
    for filename in os.listdir(in_dir):
        file_path = os.path.join(in_dir,filename)
        with open(file_path,'r',encoding='utf-8') as file:
            data = json.load(file)
            all_data.extend(data)
    json_data = json.dumps(all_data,ensure_ascii=False,indent=2)
    out_path = os.path.join(out_dir,"total.json")
    with open(out_path, 'w', encoding='utf-8') as output_file:
        output_file.write(json_data)

def kfold(origin_path,split_dir):
    """
    func: 划分训练集和测试集
    args:
        origin_path:"/home/jindongming/project/modeling/PDEval/data/dataset/10-fold/total.json"
        split_dir:"/home/jindongming/project/modeling/PDEval/data/dataset/10-fold"
    """
    with open(origin_path,'r',encoding='utf-8') as file:
        all_data = json.load(file)
    
    kf = KFold(n_splits=10,shuffle=True,random_state=42)
    for fold,(train_index,test_index) in enumerate(kf.split(all_data)):
        train_data = [all_data[i] for i in train_index]
        test_data = [all_data[i] for i in test_index]
        fold_dir = os.path.join(split_dir,f"fold_{fold}")
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        
        train_data = json.dumps(train_data,ensure_ascii=False,indent=2)
        test_data = json.dumps(test_data,ensure_ascii=False,indent=2)
        with open(os.path.join(fold_dir, 'train_data.json'), 'w',encoding='utf-8') as f:
            f.write(train_data)
        with open(os.path.join(fold_dir, 'test_data.json'), 'w',encoding='utf-8') as f:
            f.write(test_data)
    print("10-fold划分完成。")

def main():
    # in_dir = "/home/jindongming/project/modeling/PDEval/data/human_label"
    # out_dir = "/home/jindongming/project/modeling/PDEval/data/dataset/total"
    # format_human_label(in_dir,out_dir)
    # in_dir = "/home/jindongming/project/modeling/PDEval/data/dataset/total"
    # out_dir = "/home/jindongming/project/modeling/PDEval/data/dataset/10-fold"
    # merge_file(in_dir,out_dir)
    origin_path = "/home/jindongming/project/modeling/PDEval/data/dataset/10-fold/total.json"
    split_dir = "/home/jindongming/project/modeling/PDEval/data/dataset/10-fold"
    kfold(origin_path,split_dir)

if __name__=="__main__":
    main()