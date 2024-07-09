import os,sys
print(os.getcwd())
sys.path.append(os.getcwd())
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
import json
import os
from tqdm import tqdm
from src.logger import get_logger

logger = get_logger(__name__)

def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--task",type=str,required=True)
    parser.add_argument("--train_dir",type=str,required=True)
    parser.add_argument("--test_dir",type=str,required=True)
    parser.add_argument("--shot_dir",type=str,required=True)
    parser.add_argument("--shot_num",type=int,required=True)
    parser.add_argument("--prompt_dir",type=str,required=True)
    parser.add_argument("--model",type=str,required=True)
    parser.add_argument("--moda",type=str,default='greedy')
    parser.add_argument("--output_dir",type=str,required=True)
    return parser.parse_args()
    
def load_model(model_name,moda,max_tokens=512,max_model_len=2048):
    model_dir = ""
    stop_token_ids = []
    if model_name.startswith("qwen2-7b"):
        print("Loading Qwen2-7b")
        model_dir = "Qwen/Qwen2-7B"
        stop_token_ids = [151643,151644,151645]
    elif model_name.startswith("glm4-9b"):
        print("Loading glm-4-9b")
        model_dir = "THUDM/glm-4-9b"
        stop_token_ids = [151329,151330,151331,151332,151333,151334,151335,151336,151337,151338,151339,151340,151341,151342]
    elif model_name.startswith("gemma2-9b"):
        print("Loading gemma-9b")
        model_dir = "google/gemma-2-9b"
        stop_token_ids = [2,1,3,0,106,107]
    elif model_name.startswith("llama3-8b"):
        print("Loading Meta-Llama-3-8B")
        model_dir = "meta-llama/Meta-Llama-3-8B"
        stop_token_ids = [128001]
    elif model_name.startswith("mistral-7b"):
        print("Loading mistral-7b")
        model_dir = "mistralai/Mistral-7B-v0.3"
        stop_token_ids = []
    if moda=='greedy':
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=stop_token_ids,n=1)
    else:
        sampling_params = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=max_tokens, n=20)
    model = LLM(model=model_dir,tokenizer=None,max_model_len=max_model_len,trust_remote_code=True)
    # gpu_memory_utilization=0.9, tensor_parallel_size=2
    return model,sampling_params
def get_ner_shot_prompt(i,train_data,shot_data,shot_num):
    prompt = "\n## Examples\n"
    few_shots = shot_data[i]["id"][:shot_num]
    for shot_id in few_shots:
        shot_input = train_data[shot_id]["text"]
        shot_answer = train_data[shot_id]["entity"]
        prompt += f"Input:{shot_input}\n"
        prompt += f"Answer:{shot_answer}\n"
    return prompt
def get_ner_prompt(args):
    with open(args.train_dir,'r',encoding='utf-8') as train_file:
        train_data = json.load(train_file)
    with open(args.test_dir,'r',encoding='utf-8') as test_file:
        test_data = json.load(test_file)
    with open(args.shot_dir,'r',encoding='utf-8') as shot_file:
        shot_data = json.load(shot_file)
    with open(args.prompt_dir,'r',encoding='utf-8') as prompt_file:
        prompt_templeate = prompt_file.read()
    prompt_list = []
    if args.shot_num==0:
        for test_item in test_data:
            input_req = test_item["text"]
            prompt = prompt_templeate.format(examples="",input_req=input_req)
            prompt_list.append(prompt)
        return prompt_list
    else:
        shot_num = args.shot_num
        for i,test_item in enumerate(test_data):
            input_req = test_item["text"]
            shot_prompt = get_ner_shot_prompt(i,train_data,shot_data,shot_num)
            prompt = prompt_templeate.format(examples=shot_prompt,input_req=input_req)
            logger.info(prompt)
            prompt_list.append(prompt)
        return prompt_list

def get_rel_prompt(args):
    pass

def llm_inference(args,model,sampling_params):
    if args.task=="entity":
        prompt_list = get_ner_prompt(args)
    elif args.task=="relation":
        prompt_list = get_rel_prompt(args)
    
    predict_list = []
    for prompt in tqdm(prompt_list, desc='generate answer'):
        predict = model.generate([prompt],sampling_params)
        predict = predict[0].outputs[0].text
        predict_dict = {
            "predict":predict
        }
        predict_list.append(predict_dict)
    output_dir = os.path.join(args.output_dir,f"{args.task}/{args.model}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir,"fold_0")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir,f"{args.shot_num}.json")
    json_data = json.dumps(predict_list,ensure_ascii=False,indent=2)
    with open(output_path,'w',encoding='utf-8') as output_file:
        output_file.write(json_data)

def main():
    args = parser_args()
    # model,sampling_params = None,None
    model,sampling_params = load_model(args.model,args.moda)
    llm_inference(args,model,sampling_params)
    pass

if __name__=="__main__":
    main()