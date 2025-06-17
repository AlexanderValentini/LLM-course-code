# Codebase for the advanced language models course. 

This is a collection of code from exercises that could potentially be useful for a DTU course. "CMU-MiniLlama-exercise-original" contains the original code for a CMU exercise to code a small version on LLama 2 and use it for text generation. "MiniLlama-exercise-alex-solution" contains my solution for that exercise. This could be run at an exercise session

"argilla_generate_a_preference_dataset" is a tutorial about using LLM's to synthetically generate a preference dataset. It will probably not be possible to run at an exercise session, but is interesting as a proof of concept - although you would need another inference engine than the free Huggingface account. 

"unsloth_fine_tuning_llama_3" is a tutorial about fine-tuning a Llama 3 model with Unsloth, which is the fastest and most memory efficient training library for a single GPU. This is not really feasible without a GPU - and it might be a bit "high-level". Maybe its better to start with a pytorch exercise to gain a more in depth understanding. But it is certainly useful for a project.  