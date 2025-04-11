"""
Code copied and slightly modified from https://github.com/ZhaofengWu/counterfactual-evaluation/blob/master/chess/query.py

This script queries the LLM on the chess dataset and stores it results in the output directory.
It also runs the control prompts to check if the LLM understands the world it's in.
"""

import os
import pathlib


def load_data(data_file):
    return [line.strip().replace(" *", "") for line in open(data_file,)]

def templatize(mode, pgn_string, is_control=False):
    """Parse the data from the chess datasets into a prompt for the LLM"""
    if mode not in ["counter_factual", "real_world"]:
        raise ValueError(f"Invalid mode: {mode}")
    
    prompt = "You are a chess player."

    if mode == "counter_factual":
        prompt += " You are playing a chess variant where the starting positions for knights and bishops are swapped. For each color, the knights are at placed that where bishops used to be and the bishops are now placed at where knights used to be."

    if is_control:
        prompt += "Question: In this chess variant, t" if mode == "counter_factual" else "Question: T"
        prompt += f"he two {pgn_string}s on the board should be initially at which squares? Answer:"

        return prompt
    
    prompt += "Given an opening, determine whether the opening is legal. The opening doesn't need to be a good opening. Answer \"yes\" if all moves are legal. Answer \"no\" if the opening violates any rules of chess.\n"

    if mode == "real_world":
        prompt += f"Is the new opening \"{pgn_string}\" legal? "
    elif mode == "counter_factual":
        prompt += f"Under the custom variant, is the new opening \"{pgn_string}\" legal? "
  
    return prompt


def escape(str):
    return str.replace("\n", "\\n")


def main(model_name: str = "llama3-8b-instruct"):
    for exp in os.listdir("chess/data"):
        if "chess" in exp:
            data_dir = os.path.join("chess/data", exp)
            output_dir = f"chess/output/{exp}/{model_name.replace('models/', '')}_0cot{cot}"
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            for mode in ["real_world", "counter_factual"]:
                # Control experiment, should we include?
                pieces = ["white bishop", "black bishop", "white knight", "black knight"]
                templatized = [templatize(mode, piece, is_control=True) for piece in pieces]
               
                # TODO: Replace query batch funciton
                responses = query_with_hidden_states(templatized, model_name, temperature=0.1, n=15)
                output_file = os.path.join(output_dir, f"{mode}_control.txt")
                # with open(output_file, "w") as log:
                #     for piece, response_batch in zip(pieces, responses, strict=True):
                #         for response in response_batch:
                #             log.write(f"{piece} *\t{escape(response)}\n")
                # continue
                for real_world_legal in [True, False]:
                    for counter_factual_legal in [True, False]:
                        data_file = \
                            f"{data_dir}/{mode}_{'T' if real_world_legal else 'F'}_{'T' if counter_factual_legal else 'F'}.txt"

                        if not os.path.exists(data_file):
                            raise RuntimeError(f"data file {data_file} doesn't exist")

                        data = load_data(data_file)

                        templatized = [templatize(mode, pgn_string) for pgn_string in data]
                        
                        # TODO: Replace query batch funciton
                        responses = query_batch(templatized, model_name)

                        with open(output_file, "w") as log:
                            for expr, response in zip(data, responses, strict=True):
                                log.write(f"{expr} *\t{escape(response)}\n")




if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
