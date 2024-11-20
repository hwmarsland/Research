


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
from peft import PeftModel

base_model_id = "distilbert/distilbert-base-uncased"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, add_bos_token=True, trust_remote_code=True)

ft_model = PeftModel.from_pretrained(
    base_model, "florath/CoqLLM-FineTuned-Experiment-Gen0")

eval_prompt = "Lemma plus_n_O : forall n:nat, n = n + 0."

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():

    for idx in range(10):
    
        res = eval_tokenizer.decode(
            ft_model.generate(
                **model_input,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.4,
                pad_token_id=eval_tokenizer.eos_token_id,
                repetition_penalty=1.15)[0], skip_special_tokens=False)
                
        print("Result [%2d] [%s]" % (idx, res))
