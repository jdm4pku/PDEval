from transformers import AutoTokenizer, AutoModelForCausalLM


def gemma_completion(input_text,model,tokenizer):
    input_ids = tokenizer(input_text,return_tensors="pt").to("cuda:1")
    outputs = model.generate(**input_ids,max_new_tokens=512)
    print(tokenizer.decode(outputs[0]))

if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b").to("cuda:1")
    prompt = """"""
    gemma_completion(prompt,model,tokenizer)

# input_text = "Write me a poem about Machine Learning"
# input_ids = tokenizer(input_text,return_tensors="pt").to("cuda:1")

# outputs = model.generate(**input_ids,max_new_tokens=512)
# print(tokenizer.decode(outputs[0]))