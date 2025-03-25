import pandas as pd


with open("nohup.out", "r") as f:
    lines = f.readlines()

test = False
read_report = False
reset = True
results = []
for idx, line in enumerate(lines):
    if not line.strip():
        continue
    elif "evaluating test" in line:
        test = True
    elif "--" in line:
        reset = True
    elif reset:
        model_name = line.strip()
        reset = False
    elif test and "support" in line:
        read_report = True
    elif read_report:
        print(model_name)
        scores = line.split()
        report = {
            "Model": model_name,
            # "precision_0": scores[1],
            # "recall_0": scores[2],
            # "f1_0": scores[3],
            # "support_0": scores[4],
        }
        next_line = lines[idx+1]
        scores = next_line.split()
        report.update( {
            "Cheat Precision": scores[1],
            "Cheat Recall": scores[2],
            "Cheat F1-Score": scores[3],
            # "support_1": scores[4],
        } )
        next_line = lines[idx+3]
        scores = next_line.split()
        report.update( {
            # "Accuracy": scores[1],
            # "total_support": scores[2],
        } )
        next_line = lines[idx+4]
        scores = next_line.split()
        report.update( {
            # "macro_precision": scores[2],
            # "macro_recall": scores[3],
            # "Macro F1-Score": scores[4],
        } )
        next_line = lines[idx+5]
        scores = next_line.split()
        report.update( {
            # "weighted_precision": scores[2],
            # "weighted_recall": scores[3],
            # "Weighted F1-Score": scores[4],
        } )
        results.append(report)
        read_report = False
        test = False

df = pd.DataFrame(results)
df.sort_values(by="Cheat F1-Score", inplace=True)
df.to_csv('results.txt', index=False)