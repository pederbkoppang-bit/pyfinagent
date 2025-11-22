import os
import streamlit as st

print("--- Starting Secrets Debug ---")

# 1. Print the current working directory
cwd = os.getcwd()
print(f"Current Working Directory: {cwd}")

# 2. Construct the expected path for secrets.toml
expected_path = os.path.join(cwd, ".streamlit", "secrets.toml")
print(f"Streamlit is looking for secrets at: {expected_path}")

# 3. Check if the file exists at that path
if os.path.exists(expected_path):
    print("SUCCESS: secrets.toml file found at the expected path.")
    try:
        # 4. Try to access the secret to confirm it can be parsed
        project_id = st.secrets.gcp.project_id
        print(f"SUCCESS: Successfully read 'gcp.project_id': {project_id}")
    except Exception as e:
        print(f"ERROR: Found secrets.toml, but failed to read from it. The file might be malformed. Error: {e}")
else:
    print("ERROR: secrets.toml file NOT found at the expected path.")

print("--- End of Secrets Debug ---")