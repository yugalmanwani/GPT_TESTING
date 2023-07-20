from torch import cuda, bfloat16
import transformers

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

model = transformers.AutoModelForCausalLM.from_pretrained(
    'mosaicml/mpt-7b-instruct',
    trust_remote_code=True,
    torch_dtype=bfloat16,
    max_seq_len=2048
)
model.eval()
model.to(device)
print(f"Model loaded on {device}")


tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")



import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# mtp-7b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])


generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    device=device,
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=64,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)


res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])