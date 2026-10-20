#4
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text ="""
def generate_password(length=12, use_upper=True, use_digits=True, use_symbols=True):
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Password length must be a positive integer.")

    chars = string.ascii_lowercase
    if use_upper:
        chars += string.ascii_uppercase
    if use_digits:
        chars += string.digits
    if use_symbols:
        chars += "!@#$%^&*()-_=+[]{};:,.<>?/"

    if len(chars) == 0:
        raise ValueError("No characters available for password generation.")

    return ''.join(random.choice(chars) for _ in range(length))

def main():
    try:
        length = int(input("Enter password length: "))
        upper = input("Include uppercase letters? (y/n): ").strip().lower() == 'y'
        digits = input("Include digits? (y/n): ").strip().lower() == 'y'
        symbols = input("Include symbols? (y/n): ").strip().lower() == 'y'

        password = generate_password(length, upper, digits, symbols)
        print("\nGenerated Password:", password)
    except ValueError as e:
        print("Error:", e)
    except Exception as e:
        print("Unexpected error:", e)

if __name__ == "__main__":
    main()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, # JAVASCRIPT
    chunk_size=300,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)
print(len(chunks))

print(chunks[2])

#------------------------------------------------------------------

# from langchain_text_splitters import (
#     RecursiveCharacterTextSplitter,
#     Language,
# )

# text = """
# def generate_password(length=12, use_upper=True, use_digits=True, use_symbols=True):
#     if not isinstance(length, int) or length <= 0:
#         raise ValueError("Password length must be a positive integer.")

#     chars = string.ascii_lowercase
#     if use_upper:
#         chars += string.ascii_uppercase
#     if use_digits:
#         chars += string.digits
#     if use_symbols:
#         chars += "!@#$%^&*()-_=+[]{};:,.<>?/"

#     if len(chars) == 0:
#         raise ValueError("No characters available for password generation.")

#     return ''.join(random.choice(chars) for _ in range(length))

# def main():
#     try:
#         length = int(input("Enter password length: "))
#         upper = input("Include uppercase letters? (y/n): ").strip().lower() == 'y'
#         digits = input("Include digits? (y/n): ").strip().lower() == 'y'
#         symbols = input("Include symbols? (y/n): ").strip().lower() == 'y'

#         password = generate_password(length, upper, digits, symbols)
#         print("\\nGenerated Password:", password)

#     except ValueError as e:
#         print("Error:", e)
#     except Exception as e:
#         print("Unexpected error:", e)

# if __name__ == "__main__":
#     main()
# """

# splitter = RecursiveCharacterTextSplitter.from_language(
#     language=Language.PYTHON,
#     chunk_size=300,
#     chunk_overlap=0,
# )

# chunks = splitter.split_text(text)

# print("Number of chunks:", len(chunks))

# for i, chunk in enumerate(chunks):
#     print(f"\n--- Chunk {i} ---")
#     print(chunk)