import json

path = r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\objective4_live_metrics_summary.json"
with open(path, encoding="utf-8") as f:
    d = json.load(f)

for m, val in d.items():
    print(f"\n{'='*55}\n{m} Live Metrics in DOM\n{'='*55}")
    for met in val["metrics"]:
        l = met["label"].upper()
        if any(k in l for k in ["CHANGED PIXELS", "CHANGE RATIO", "FOOTPRINT", "PRECISION", "RECALL", "SPECIFICITY", "IOU", "DICE"]):
            print(f"  {met['label']}: {met['val']}")
    
    # Also parse confusion matrix text
    m_str = val.get("matrixStr", "")
    for line in m_str.split("\n"):
        if any(w in line for w in ["True Positives", "False Positives", "False Negatives", "True Negatives", "Total Valid"]):
            print(f"  [Matrix] {line.strip()}")
