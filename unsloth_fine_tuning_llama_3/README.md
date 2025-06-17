# Fine-tuning Llama 3.2 with Unsloth. 

This is my implementation of the tutorial by Maxime Labonne:

https://huggingface.co/blog/mlabonne/sft-llama3

Here we fine-tune a Llama 3 model with Unsloth, which is the fastest and most memory efficient training library for a single GPU. 

It is possible to install unsloth on both windows or WSL. The guide in the tutorial is outdated - so follow this link instead: 

https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation

Note that you will also need to install Triton (preferably in the same requirements file). Please follow this link:

https://github.com/woct0rdho/triton-windows

Here it is important to match the torch+cuda version with the correct triton version - otherwise you will get incompatibility errors.

It is also important to install the newest NVIDIA GPU drivers - otherwise you will get "RuntimeError: PassManager::run failed". Even though my drivers were only a couple of months old I got an error.

Note that running this is not really feasible without a GPU - and it might be a bit "high-level". Maybe its better to start with a pytorch exercise to gain a more in depth understanding. But it is certainly useful for a project.

The original model took 13 hours to run on a GTX 4080 Super GPU or 48 hours on a GTX 2080 laptop gpu. After changing the model to the 1B version, the runtime was reduced to 2 hours on the GTX 4080 or 12 hours on the  Nvidia 2080 gpu. This might be further reduced by training on a smaller subset of the data.