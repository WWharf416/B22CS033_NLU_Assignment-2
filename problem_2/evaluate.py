import os

def calculate_metrics(generated_file, training_set):
    if not os.path.exists(generated_file):
        print(f"File {generated_file} not found. Did you run the training script?")
        return
        
    with open(generated_file, "r", encoding="utf-8") as f:
        generated_names = [line.strip() for line in f if line.strip()]
        
    total_generated = len(generated_names)
    if total_generated == 0:
        print(f"No names found in {generated_file}.")
        return

    # 1. Novelty Rate: % not in training set
    novel_names = [name for name in generated_names if name not in training_set]
    novelty_rate = (len(novel_names) / total_generated) * 100

    # 2. Diversity: Unique generated / total generated
    unique_names = set(generated_names)
    diversity = (len(unique_names) / total_generated) * 100

    print(f"--- Metrics for {generated_file} ---")
    print(f"Total Names Generated: {total_generated}")
    print(f"Novelty Rate: {novelty_rate:.2f}%")
    print(f"Diversity:    {diversity:.2f}%\n")


if __name__ == "__main__":
    # Load original training names into a set for fast lookup
    if not os.path.exists("names.txt"):
        print("Original names.txt not found. Cannot compute novelty.")
    else:
        with open("names.txt", "r", encoding="utf-8") as f:
            training_names = set([line.strip().lower() for line in f if line.strip()])
        
        print("EVALUATION RESULTS\n" + "="*20)
        calculate_metrics("generated_rnn.txt", training_names)
        calculate_metrics("generated_blstm.txt", training_names)
        calculate_metrics("generated_attn.txt", training_names)