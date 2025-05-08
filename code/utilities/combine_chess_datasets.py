"""Code to combine the chess text files into a single JSONL file."""

import os
import glob
import json

def load_text_files(input_dir, output_file="chess_data.jsonl"):
    """
    Load the text files from the input directory and create a JSONL file.
    
    Each line in the output JSONL file will be a separate JSON object.
    """
    text_files = glob.glob(os.path.join(input_dir, "*.txt"))
    data = []
    
    print(f"Found {len(text_files)} text files to process")
    
    # Process each file
    for file_path in text_files:
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")
        
        # Determine the mode (counter_factual or real_world)
        if 'counter_factual' in file_name:
            mode = 'counter_factual'
        elif 'real_world' in file_name:
            mode = 'real_world'
        else:
            print(f"Skipping unknown file type: {file_name}")
            continue
        
        # Determine the answers based on the filename (T_F or F_T)
        parts = file_name.split('_')
        if len(parts) < 3:
            print(f"Skipping file with invalid name format: {file_name}")
            continue
            
        if parts[2] == 'F':
            real_world_answer = False
            counter_factual_answer = True
        elif parts[2] == 'T':
            real_world_answer = True
            counter_factual_answer = False
        else:
            print(f"Skipping file with unknown answer format: {file_name}")
            continue
        
        # Read the file and process each line
        with open(file_path, "r") as f:
            lines = f.readlines()
            print(f"  Found {len(lines)} chess openings")
            
            for line in lines:
                line = line.strip()
                # Remove the trailing asterisk if present
                if line.endswith("*"):
                    line = line[:-1].strip()
                
                if line:  # Skip empty lines
                    # Create a JSON object for this opening
                    entry = {
                        "mode": mode,
                        "real_world_answer": real_world_answer,
                        "counter_factual_answer": counter_factual_answer,
                        "opening": line
                    }
                    data.append(entry)
    
    # Save the data as a JSONL file (one JSON object per line)
    output_path = os.path.join(input_dir, output_file)
    with open(output_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Successfully created {output_path} with {len(data)} entries")
    return data


if __name__ == "__main__":
    # Process the chess data files
    data = load_text_files("/home/jholshuijsen/reasoning-reciting-probing/inputs/chess/data/")
    print(f"Processed {len(data)} total chess openings")



