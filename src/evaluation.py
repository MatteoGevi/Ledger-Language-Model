from sklearn.metrics import accuracy_score

def evaluate_agent(predicted_entries, ground_truth_entries):
    correct = 0
    total = len(ground_truth_entries)

    for pred, truth in zip(predicted_entries, ground_truth_entries):
        if pred == truth:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy