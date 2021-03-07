import pandas as pd, sys

pickle_file = sys.argv[1]
name_of_output_file = sys.argv[2]
def main():
    read_pickle_file(pickle_file)
    
def read_pickle_file(pickle_file):
    pickle_data = pd.read_pickle(pickle_file)
    return pickle_data

if __name__ == "__main__":
    main()