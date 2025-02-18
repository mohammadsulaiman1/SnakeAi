import pickle

# Load memory
with open("memory.pkl", "rb") as f:
    memory = pickle.load(f)

# Check the type of memory
print(f"✅ Memory loaded successfully. Type: {type(memory)}")

# If it's a deque (which it should be), convert to list
if isinstance(memory, (list, deque)):
    print(f"✅ Memory contains {len(memory)} experiences.")
    print("🔹 Sample memory (first 5 entries):", list(memory)[:5])
else:
    print("⚠️ Warning: Memory format is unexpected.")
    print(memory)
