from sanic import Sanic, response
from sanic.response import json as json_response
from warmup import load_model
from transformers import AutoTokenizer
from run import run_model
# do the warmup step globally, to have a reuseable model instance
model = load_model()
tokenizer = AutoTokenizer.from_pretrained("t5-small")
app = Sanic("my_app")

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
    
    output = run_model(model,tokenizer,prompt)
    response = {"output": output}
    return json_response(response) # Do not edit - returning a dictionary as JSON is a required interface


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000", workers=1)
