# debug_bart_path.py

import sys

print("--- Starting BART Environment Diagnosis ---")

# Print the Python version and executable path being used
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

# Print the paths Python searches for modules
print("\nPython's Search Paths (sys.path):")
for path in sys.path:
    print(f"  - {path}")

try:
    # We will attempt the import and then inspect the result
    import bart
    print("\n--- Import Analysis ---")
    print("SUCCESS: 'import bart' command executed.")

    # 1. This is the most important line. It tells us WHICH file was loaded.
    print(f"\nLocation of the imported 'bart' module (__file__): {bart.__file__}")

    # 2. Let's see what is inside the module object that was loaded.
    print(f"\nContents of the 'bart' module (dir(bart)): {dir(bart)}")
    
    # 3. Let's specifically check for the 'bart' function inside the module.
    has_bart_function = hasattr(bart, 'bart')
    print(f"\nDoes the module have a function named 'bart'? {has_bart_function}")

    if has_bart_function:
        print("  - Diagnosis: This is correct. The call should be 'bart.bart(...)'.")
    else:
        print("  - Diagnosis: CRITICAL ERROR! The loaded module does NOT contain the 'bart' function. This is the source of the problem.")

except ImportError as e:
    print(f"\nCRITICAL ERROR: Failed to import bart. Error: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

print("\n--- Diagnosis Complete ---")