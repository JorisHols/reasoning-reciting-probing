import os
import argparse
from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser(description="Inspect a Hugging Face dataset and print llm_answer for each row")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("dataset_type", type=str, help="Type of dataset to inspect", choices=["probe", "intervention"])
    args = parser.parse_args()
    
    # Check if the path exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist")
        return
    
    try:
        # Load the dataset from disk
        dataset = load_from_disk(args.dataset_path)
        print(f"Loaded dataset with {len(dataset)} rows")
        
        # Check if llm_response exists in the dataset
        if args.dataset_type == "probe" and "llm_response" not in dataset.column_names:
            print("Error: Dataset does not contain 'llm_response' column")
            print(f"Available columns: {dataset.column_names}")
            return
        elif args.dataset_type == "intervention" and "intervention_response" not in dataset.column_names:
            print("Error: Dataset does not contain 'intervention_response' column")
            print(f"Available columns: {dataset.column_names}")
            return
        
        output_file = f"results/{args.dataset_path.split('/')[-1]}.txt"
        # Print each llm_response on a new line
        for i, row in enumerate(dataset):
            if args.dataset_type == "probe":
                print(f"Row {i} llm_response:")
                print(row["llm_response"])
            elif args.dataset_type == "intervention":
                print(f"Row {i} intervention_response:")
                print(row["intervention_response"])
            print("-" * 80)  # Separator between responses
            with open(output_file, "a") as f:
                f.write(f"Row {i} intervention_response: {row['intervention_response']}\n")
            
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")



if __name__ == "__main__":
    main()
