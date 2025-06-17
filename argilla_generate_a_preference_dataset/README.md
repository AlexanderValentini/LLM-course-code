# Codebase for the advanced language models course. 

This is an adaptation of the tutorial from this link: 

https://distilabel.argilla.io/dev/sections/pipeline_samples/tutorials/generate_preference_dataset/#run-the-pipeline

The task is to use LLM's to synthetically generate a preference dataset. 

Some patches are necessary to make the code run. First I am using a PatchedInferenceEndpointsLLM, since theoriginal InferenceEndpointsLLM will try to access the paid Hugging Face Inference Endpoints API, which is not available in the free tier (as far as I understood). It will return None as the model. Also it will run a status check of the model and cause an error. 

Also I am using the dotenv library to load my HF api token stored in a .env file. 

To get the tutorial to work, first you will need an account at Huggingface and create a new access token (under "Access Tokens"):

![](image-1.png)

![](image.png)

It will probably not be possible to run at an exercise session, but is interesting as a proof of concept - although you would need another inference engine than the free Huggingface account. 