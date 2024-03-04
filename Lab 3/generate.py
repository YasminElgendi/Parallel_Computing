
import argparse
import random

def generate_test_file(size,filename):
    with open(filename, "w") as file:
        for _ in range(size):
            file.write(str(round(random.uniform(1, 10), 1))+ "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate input file.")
    parser.add_argument("size", type=int, help="number of elements")
    parser.add_argument("filename", type=str, help="Output filename")
    args = parser.parse_args()

    generate_test_file(args.size,args.filename)

if __name__ == "__main__":
    main()


# python ./tests/generate.py num_test_cases rows columns filename
#  python ./tests/generate.py 1 4 3 ./tests/test_4_3.txt