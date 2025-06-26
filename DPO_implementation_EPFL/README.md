# Implementation of DPO for an EPFL project.

These scripts contain my implementation for DPO and MCQA evaluation as a part of a project I made at EPFL.

model_base.py is not relevant. It is necessary to run the evaluation and contains some helper functions created by the TA's at EPFL. 

evaluator.py is running the main evaluation of the model. Either it evaluates accuracy on a multiple-choice dataset or the model's assigned rewards on a a preference dataset. This was also not created by me. 

For a reference implementation of DPO evaluation, see the model_dpo.py script. Look under the "AutoDPOModelForCausalLM" class and specifically at the get_logprobs and prediction_step_reward functions. These compute the rewards of the rejected and chosen samples (comparing the ground truth and model generations). 

Note that this might take too much time for one exercise and does not represent the whole training process, but it should mostly be seen as an inspiration for a potential exercise about DPO (namely to implement the formulas from the paper).