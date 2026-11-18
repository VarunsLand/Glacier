import json

path = r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\objective4_five_methods_live_audit.json"
with open(path, encoding="utf-8") as f:
    d = json.load(f)

for m_key, m_val in d["methods"].items():
    print(f"\n{'='*55}\n{m_key}: {m_val['prefix']}\n{'='*55}")
    panels = m_val["state"]["panels"]
    for p in panels[7:13]:
        p_num = p["index"] - 7 + 1
        print(f"  Panel {p_num}: {p['caption']} | Media ID: {p['src_prefix'][-12:]}")
