import sys

# compre two files
def main():
    if len(sys.argv) != 3:
        print("Usage: python compare.py file1 file2")
        return
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    with open(file1, 'r') as f1:
        with open(file2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            if len(lines1) != len(lines2):
                print("Files are different")
                return
            for i in range(len(lines1)):
                if lines1[i] != lines2[i]:
                    print("Files are different")
                    return
                
    print("Files are the same")


if __name__ == "__main__":
    main()
