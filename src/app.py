from sanic import Sanic, response
from warmup import load_model
from transformers import AutoTokenizer
# do the warmup step globally, to have a reuseable model instance
model = load_model()
tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
app = Sanic("my_app")

def run_model(input,max_length=None,min_length=None,temp=0.9,topP=0.9,topK=50):
    #move the input tokens to the first gpu
    input_tokens = tokenizer.encode(input, return_tensors="pt").to("cuda:0")
    
    output = model.generate(input_tokens,do_sample=True,max_length=max_length,min_length=min_length,temperature=temp,top_p=topP,top_k = topK)

    res = tokenizer.batch_decode(output, skip_special_tokens=True)
    return res


@app.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    return response.json({"state": "healthy"})

@app.route('/', methods=["POST"]) # Do not edit - POST requests to "/" are a required interface
def inference(request):
    try:
        model_inputs = response.json.loads(request.json)
    except:
        model_inputs = request.json

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return response.json({'message': "No prompt provided"})
    
    output = run_model(prompt)
    response = {"output": output}
    return response.json(output) # Do not edit - returning a dictionary as JSON is a required interface


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000", workers=1)