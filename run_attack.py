import config
import json
from helper import load_model_and_tokenizer, get_tokens, parse_csv, get_model_path
from behavior import Behavior, RRBehavior
import argparse
import pgd, softprompt, rr
import warnings


def main(input_file, output_file, model, attack_name, num_behaviors, behavior_start=0):
    attack= attack_name
    prefix = ""
    suffix = ""
    model_path = get_model_path(model)
    if model_path == config.vicuna_path or model_path == config.mistral_path:
        suffix = ":"
    import torch
    torch.manual_seed(config.seed)
    model, tokenizer = load_model_and_tokenizer(model_path)
    rows = parse_csv(input_file)

    if attack == "RR":
        attackconfig = config.rrconfig
        checkpoint_files = [output_file.replace('.jsonl', f'_{str(checkpoint).replace("0.", "")}.jsonl') for checkpoint in rrconfig.checkpoints]
    elif attack == "SoftPrompt":
        attackconfig = ""
    elif attack == "PGD":
        attackconfig = config.pgdconfig

    for row in rows[behavior_start:behavior_start+num_behaviors]:
        if len(row) != 2:
            continue
        user_prompt, target = row
        print(row)
        try:
            messages = prefix + user_prompt + suffix
            if attack == "RR":
                result = rr.run(model, tokenizer, messages, target, attackconfig)
                for i, checkpoint in enumerate(attackconfig.checkpoints):
                    best_string = result.best_strings[i]
                    mean_distance = result.distances[i]
                    
                    final_string = prefix + user_prompt + suffix + best_string
                    final_string_ids = get_tokens(final_string, tokenizer, config.device)
                    attention_mask = torch.ones(final_string_ids.shape, device=config.device)
                    
                    # Generate the output based on the final string
                    generated_output = model.generate(
                        final_string_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        max_length=200,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False,
                        top_p=None,
                        top_k=None,
                        temperature=None
                    )
                    generated_output_string = tokenizer.decode(generated_output[0], skip_special_tokens=True)[len(final_string):]

                    # Create the Behavior object
                    behavior = RRBehavior(user_prompt, best_string, generated_output_string, "", "", mean_distance)
                    
                    # Write to the corresponding checkpoint output file
                    with open(checkpoint_files[i], 'a') as f:
                        f.write(json.dumps(behavior.to_dict()) + '\n')
                    f.close()

                    # Optionally write to a dump file for each checkpoint
                    # dump_file = output_file.replace('.jsonl', f'_dump_{str(checkpoint).replace(".", "")}.jsonl')
                    # with open(dump_file, 'a') as d:
                    #     d.write(json.dumps(result.to_dict()) + '\n')
                    # d.close()
            else:
                if attack == "SoftPrompt":
                    result = softprompt.run(model, tokenizer, messages, target, attackconfig)
                elif attack == "PGD":
                    result = pgd.run(model, tokenizer, messages, target, attackconfig)
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
        if attack != "RR":
            final_string = prefix + user_prompt + suffix + result.best_string
            # Suppress specific warnings
            warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`. However, `temperature` is set to.*")
            warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`. However, `top_p` is set to.*")

            final_string_ids = get_tokens(final_string, tokenizer, config.device)
            attention_mask = torch.ones(final_string_ids.shape, device=config.device)
            generated_output = model.generate(
                final_string_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                max_length=200,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None
            )
            generated_output_string = tokenizer.decode(generated_output[0], skip_special_tokens = True)[len(final_string):]
            # print(generated_output_string)
            behavior = Behavior(user_prompt, result.best_string, generated_output_string, "", "")
            
            with open(output_file, 'a') as f:
                f.write(json.dumps(behavior.to_dict()) + '\n')
            f.close()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process command line arguments')
    parser.add_argument('--input_file', type=str, help='Path to the dataset file')
    parser.add_argument('--output_file', type=str, help='Specify output file name')
    parser.add_argument("--model", type=str, help="Name of the model")
    parser.add_argument("--attack_name", type=str, help="Name of the attack method")
    parser.add_argument("--num_behaviors", type=int, help="Number of harmful behaviors")
    parser.add_argument("--behavior_start", type=int, help="Starting index for behaviors", required=False, default=0)

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.model, args.attack_name, args.num_behaviors, args.behavior_start)