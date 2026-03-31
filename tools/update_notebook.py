import json

nb_path = "/home/vsc/LLM_TUNE/structure-stability/code/Regularization_v2.9.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])

        # 1. Update extract_all_features
        if "def extract_all_features" in source:
            source = source.replace(
                "video_path = TRAIN_ROOT / sample_id / 'simulation.mp4'",
                "sample_dir = row.get('sample_dir', TRAIN_ROOT)\n        video_path = Path(sample_dir) / sample_id / 'simulation.mp4'"
            )
            
        # 2. Add extract call for simulation data
        if "meta = extract_all_features(train_df)" in source:
            # We ONLY add it if it's not already added
            if "sim_summary_path = SIMULATION_DATA_DIR" not in source:
                addition = """
sim_summary_path = SIMULATION_DATA_DIR / 'generated_summary.json'
if sim_summary_path.exists():
    import json
    with open(sim_summary_path, 'r', encoding='utf-8') as f:
        sim_meta = json.load(f)
    sim_rows = [{'id': item['id'], 'label': item['label'], 'sample_dir': str(SIMULATION_DATA_DIR)} for item in sim_meta]
    sim_df = pd.DataFrame(sim_rows)
    print(f"시뮬레이션 데이터: {len(sim_df)}개")
    extract_all_features(sim_df)
"""
                source = source + addition

        # 3. Add to train_df_copy
        if "print('Final train class ratio:')" in source and "train_df_copy = pd.concat" in source:
            if "sim_summary_path = SIMULATION_DATA_DIR" not in source:
                addition = """
sim_summary_path = SIMULATION_DATA_DIR / 'generated_summary.json'
if sim_summary_path.exists():
    import json
    with open(sim_summary_path, 'r', encoding='utf-8') as f:
        sim_meta = json.load(f)
    sim_rows = [{'id': item['id'], 'label': item['label'], 'sample_dir': str(SIMULATION_DATA_DIR)} for item in sim_meta]
    sim_df = pd.DataFrame(sim_rows)
    train_df_copy = pd.concat([train_df_copy, sim_df], ignore_index=True)
    print(f'simulation data added: {len(sim_df)} samples')

"""
                source = source.replace("print('Final train class ratio:')", addition + "print('Final train class ratio:')")

        # Put back
        lines = source.split("\n")
        cell["source"] = [lines[i] + "\n" for i in range(len(lines) - 1)] + [lines[-1]]
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"].pop()

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
