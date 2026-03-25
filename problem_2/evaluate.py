import os

def load_names(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

def evaluate_model(train_file, generated_file):
    train_names = set(load_names(train_file))
    gen_names = load_names(generated_file)
    
    total_generated = len(gen_names)
    if total_generated == 0:
        return 0.0, 0.0
        
    unique_generated = set(gen_names)
    novel_names = [name for name in gen_names if name not in train_names]
    
    novelty_rate = (len(novel_names) / total_generated) * 100
    diversity_rate = (len(unique_generated) / total_generated) * 100
    
    return novelty_rate, diversity_rate

if __name__ == "__main__":
    train_file = "names.txt"
    models = ["generated_rnn.txt", "generated_blstm.txt", "generated_attn.txt"]
    
    print(f"{'Model':<25} | {'Novelty Rate (%)':<18} | {'Diversity Rate (%)'}")
    print("-" * 65)
    
    for gen_file in models:
        nov, div = evaluate_model(train_file, gen_file)
        if nov == 0.0 and div == 0.0:
            print(f"{gen_file:<25} | File not found     | -")
        else:
            print(f"{gen_file:<25} | {nov:<18.2f} | {div:.2f}")