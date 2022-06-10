def run_model(model,tokenizer,input,max_length=None,min_length=None,temp=0.9,topP=0.9,topK=50):
    #move the input tokens to the first gpu
    input_tokens = tokenizer.encode(input, return_tensors="pt").to("cuda:0")
    
    output = model.generate(input_tokens,do_sample=True,max_length=max_length,min_length=min_length,temperature=temp,top_p=topP,top_k = topK)

    res = tokenizer.batch_decode(output, skip_special_tokens=True)
    return res[0]
