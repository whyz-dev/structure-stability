import json

nb_path = "/home/vsc/LLM_TUNE/structure-stability/code/Knowledge_Distillation_v3.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

changed = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for i, line in enumerate(cell["source"]):
            if "loss_feature =" in line or "loss_feature=" in line or "feature_loss =" in line:
                if "student_feat" in line and "student_feats" not in line:
                    cell["source"][i] = line.replace("student_feat", "student_feats")
                    changed = True
                if "teacher_feat" in line and "teacher_feats" not in line:
                    cell["source"][i] = line.replace("teacher_feat", "teacher_feats")
                    changed = True

if changed:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook fixed successfully.")
else:
    print("String not found or no changes needed.")
