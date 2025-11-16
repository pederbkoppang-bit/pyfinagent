import os
import sys

# This script is designed to be run from your app's directory.
# It will locate the problematic file in your Python environment and patch it.

try:
    # Find the site-packages directory for the current Python interpreter
    site_packages = next(p for p in sys.path if 'site-packages' in p)
    
    # Construct the full path to the file that needs patching
    file_to_patch = os.path.join(site_packages, 'bs4', '__init__.py')

    if not os.path.exists(file_to_patch):
        print(f"Error: Could not find the file to patch at {file_to_patch}")
        sys.exit(1)

    # Read the content of the file
    with open(file_to_patch, 'r', encoding='utf-8') as f:
        content = f.read()

    # Define the incorrect and correct lines
    incorrect_line = "from . import builder"
    correct_line = "from .builder import builder_registry"

    # Check if the patch is needed and apply it if it is
    if incorrect_line in content:
        print(f"Found incorrect import in {file_to_patch}. Patching file...")
        new_content = content.replace(incorrect_line, correct_line)
        
        # Write the corrected content back to the file
        with open(file_to_patch, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Patch applied successfully!")
    else:
        print("File already appears to be patched or does not contain the incorrect import. No action taken.")

except StopIteration:
    print("Error: Could not find the 'site-packages' directory in your Python path.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

