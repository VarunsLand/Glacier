import ast
path = r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\src\ui\visual_monitoring.py"
with open(path, encoding="utf-8") as f:
    src = f.read()
ast.parse(src)
print("Syntax OK")
