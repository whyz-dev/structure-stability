import json

nb_path = "/home/vsc/LLM_TUNE/structure-stability/code/Knowledge_Distillation_v3.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

changed = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for i, line in enumerate(cell["source"]):
            if "return out, student_feat" in line:
                cell["source"][i] = line.replace("student_feat", "fused")
                changed = True

if changed:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook fixed successfully.")
else:
    print("String not found or no changes needed.")
