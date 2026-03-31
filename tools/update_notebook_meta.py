import json

nb_path = "/home/vsc/LLM_TUNE/structure-stability/code/Regularization_v2.9.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

old_code = """sim_summary_path = SIMULATION_DATA_DIR / 'generated_summary.json'
if sim_summary_path.exists():
    import json
    with open(sim_summary_path, 'r', encoding='utf-8') as f:
        sim_meta = json.load(f)
    sim_rows = [{'id': item['id'], 'label': item['label'], 'sample_dir': str(SIMULATION_DATA_DIR)} for item in sim_meta]
    sim_df = pd.DataFrame(sim_rows)"""

new_code = """sim_rows = []
for sim_dir in SIMULATION_DATA_DIR.glob('SIM20_*'):
    if sim_dir.is_dir():
        meta_path = sim_dir / 'meta.json'
        if meta_path.exists():
            import json
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            sim_rows.append({'id': meta['id'], 'label': meta['label'], 'sample_dir': str(SIMULATION_DATA_DIR)})
if sim_rows:
    sim_df = pd.DataFrame(sim_rows)"""

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if old_code in source:
            source = source.replace(old_code, new_code)
            lines = source.split("\n")
            cell["source"] = [lines[i] + "\n" for i in range(len(lines) - 1)] + [lines[-1]]
            if cell["source"] and cell["source"][-1] == "\n":
                cell["source"].pop()

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook meta loop updated successfully.")
