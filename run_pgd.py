import config
import json
from helper import load_model_and_tokenizer, get_tokens, parse_csv, get_model_path
from transformers import LlamaForCausalLM
from behavior import Behavior
import torch
import argparse
import pgd
import warnings


def main(input_file, output_file, model, dataset_name, num_behaviors, behavior_start=0):
    model_path = get_model_path(model)
    model, tokenizer = load_model_and_tokenizer(model_path)
    pgdconfig = config.pgdconfig
    rows = parse_csv(input_file)
    for row in rows[behavior_start:behavior_start+num_behaviors]:
        if len(row) != 2:
            continue
        user_prompt, target = row
        print(row)
        try:
            messages = user_prompt
            # if isinstance(model, LlamaForCausalLM):
            #     messages = user_prompt
            # else:
            #     messages = [{"role": "user", "content": user_prompt}]
            result = pgd.run(model, tokenizer, messages, target, pgdconfig)
            # continue
            dump_file = output_file.replace('.jsonl', '_dump.jsonl')
            with open(dump_file, 'a') as d:
                d.write(json.dumps(result.__dict__) + '\n')
            d.close()
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("CUDA out of memory. Clearing cache and retrying...")
                torch.cuda.empty_cache()
            else:
                raise e
        final_string = user_prompt + " " + result.best_string
        # Suppress specific warnings
        warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`. However, `temperature` is set to.*")
        warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`. However, `top_p` is set to.*")

        if isinstance(model, LlamaForCausalLM):
            final_string_ids = get_tokens(final_string, tokenizer, config.device)
            generated_output = model.generate(
                final_string_ids.unsqueeze(0),
                max_length=200,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
                )
            generated_output_string = tokenizer.decode(generated_output[0][1:].cpu().numpy()).strip()[len(final_string):]

        else:
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            messages[-1]["content"] = final_string
            input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            attention_mask = (input != tokenizer.pad_token_id).long()
            output = model.generate(input, do_sample=False, max_new_tokens=200, attention_mask=attention_mask)
            generated_output_string = tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]
        
        behavior = Behavior(user_prompt, result.best_string, generated_output_string, "", "")
        
        with open(output_file, 'a') as f:
            f.write(json.dumps(behavior.to_dict()) + '\n')
        f.close()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process command line arguments')
    parser.add_argument('--input_file', type=str, help='Path to the dataset file')
    parser.add_argument('--output_file', type=str, help='Specify output file name')
    parser.add_argument("--model", type=str, help="Name of the model")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--num_behaviors", type=int, help="Number of harmful behaviors")
    parser.add_argument("--behavior_start", type=int, help="Starting index for behaviors", required=False, default=0)

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.model, args.dataset_name, args.num_behaviors, args.behavior_start)