from PIL import Image
from pathlib import Path

ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")
figs = ROOT / "outputs" / "figures"

for name in ["rhone", "great_aletsch", "unteraar", "gorner", "findel"]:
    p = figs / f"glims_boundary_{name}.png"
    if p.exists():
        im = Image.open(p)
        print(f"{name}: size={im.size}, mode={im.mode}")
