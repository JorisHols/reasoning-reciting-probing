base = 10
type = "probe"

# Probing with interventino settings
alpha = 0.15
layer = 9

path_to_chunks_folder = f"./outputs/arithmetic/{type}/base{base}/with_intervention/intervention_probe_base{base}_alpha_{alpha:.2f}_layer_dofm_{layer}/chunks"

if __name__ == "__main__":
    import subprocess
    import sys
    import os

    print("Setting up Poetry environment")
    os.chdir('/gpfs/home5/jholshuijsen/reasoning-reciting-probing')
    
    # Configure Poetry to create venv in project
    subprocess.run(['poetry', 'config', 'virtualenvs.in-project', 'true', '--local'], check=True)

    # Install dependencies
    print("Installing dependencies...")
    subprocess.run(['poetry', 'install', '--no-interaction', '--no-root'], check=True)
    
    # Get the virtual environment path
    result = subprocess.run(['poetry', 'env', 'info', '--path'], 
                           capture_output=True, text=True, check=True)
    venv_path = result.stdout.strip()
    
    # Activate the virtual environment by modifying sys.path
    venv_site_packages = os.path.join(venv_path, 'lib', 
                                     f'python{sys.version_info.major}.{sys.version_info.minor}', 
                                     'site-packages')
    sys.path.insert(0, venv_site_packages)
    
    print(f"Poetry virtual environment activated at: {venv_path}")

    from datasets import load_from_disk, concatenate_datasets
    import os

    # List all folders in the path
    folders = os.listdir(path_to_chunks_folder)

    # Sort folders by number
    folders.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    output_dir = f'./results/arithmetic/{type}/base{base}/with_intervention/alpha_{alpha:.2f}_layer_dofm_{layer}'
    os.makedirs(output_dir, exist_ok=True)

    combined_dataset = None
    for folder in folders:
        # Load the dataset
        dataset = load_from_disk(os.path.join(path_to_chunks_folder, folder))
        # Concatenate the dataset
        if combined_dataset is None:
            combined_dataset = dataset
        else:
            combined_dataset = concatenate_datasets([combined_dataset, dataset])

    # Save the combined dataset
    combined_dataset.save_to_disk(os.path.join(output_dir, "combined"))




