from datasets import load_dataset


def prepare_dataset(dataset_name):
    """
    Prepare a dataset for IRL training based on its name
    
    Args:
        dataset_name: Name of the dataset to load ("limo" or "s1k")
        
    Returns:
        dataset: Loaded dataset
    """

    data = []
    if dataset_name.lower() == "limo":
        # Load LIMO dataset
        dataset = load_dataset("GAIR/LIMO", split="train")
        for i in range(len(dataset)):
            data.append({"question": dataset[i]["question"], "response": dataset[i]["solution"], "answer": dataset[i]["answer"]})
    
    elif dataset_name.lower() == "s1k":
        # Load s1K-1.1 dataset
        dataset = load_dataset("simplescaling/s1K-1.1", split="train")
        for i in range(len(dataset)):
            data.append({"question": dataset[i]["question"], "response": dataset[i]["gemini_thinking_trajectory"] + "\n" + dataset[i]["gemini_attempt"], "answer": dataset[i]["solution"]})
    
    return data