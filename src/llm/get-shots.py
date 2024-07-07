import argparse
import os
from tqdm import tqdm
import json
import faiss
from simcse import SimCSE
from sklearn.metrics.pairwise import cosine_similarity

def read_sentence(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        data = json.load(file)
    sentences = [item["text"] for item in data] 
    return sentences

def do_similarity_search(in_dir,out_dir,k):
    all_data = []
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    test_path = os.path.join(in_dir,"test_data.json")
    train_path = os.path.join(in_dir,"train_data.json")
    test_sentence = read_sentence(test_path)
    train_sentence = read_sentence(train_path)
    # 计算嵌入
    test_embedding = model.encode(test_sentence)
    train_embedding = model.encode(train_sentence)
    # 创建索引
    dimension = train_embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(train_embedding)
    distances,indices = index.search(test_embedding,k)
    # for i,sentence in enumerate(test_sentence):
    for i in tqdm(range(len(test_sentence)), desc='Processing sentences'):
        sentence = test_sentence[i]
        text = sentence
        id = indices[i].tolist()
        example = [train_sentence[idx] for idx in indices[i]]
        item = {
            "text":text,
            "id":id,
            "example":example
        }
        all_data.append(item)
    json_data = json.dumps(all_data,ensure_ascii=False,indent=2)
    out_path = os.path.join(out_dir,"shot.json")
    with open(out_path, 'w', encoding='utf-8') as output_file:
        output_file.write(json_data)
def main():
    in_dir = "/home/jindongming/project/modeling/PDEval/data/dataset/10-fold/fold_9"
    out_dir = "/home/jindongming/project/modeling/PDEval/input/fold_9"
    k = 10 # 检索的数量
    do_similarity_search(in_dir,out_dir,k)


if __name__=="__main__":
    main()