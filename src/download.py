# In this file, we define download_model
# It runs during container build time to get model weights locally

# In this example: A Huggingface BERT model

from transformers import T5ForConditionalGeneration

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

if __name__ == "__main__":
    download_model()
