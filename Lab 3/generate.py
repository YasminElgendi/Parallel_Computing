
import argparse
import random

def generate_test_file(array_size,filename):
    with open(filename, "w") as file:
        for i in range(array_size):
            line = str(round(random.uniform(1, 10), 1))
            line += '\n' if i != array_size - 1 else ''
            file.write(line)

def main():
    parser = argparse.ArgumentParser(description="Generate input file.")
    parser.add_argument("array_size", type=int, help="number of elements")
    parser.add_argument("filename", type=str, help="Output filename")
    args = parser.parse_args()

    generate_test_file(args.array_size,args.filename)

if __name__ == "__main__":
    main()


# python ./tests/generate.py array_array_size filename
#  python ./tests/generate.py 10 ./tests/test_10.txt