import argparse

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Print the input argument to the console.")
    
    # Add an argument
    parser.add_argument("input_argument", type=str, help="The input argument to be printed")
    
    # Parse the arguments
    args = parser.parse_args()

    # Get the file path from the input argument instead of user input
    img_path = args.input_argument
    
    # Print the input argument (DELETE)
    print(f"Input argument: {args.input_argument}")
    print("success")

    # Ab hier text.py weiterfuehren ...

if __name__ == "__main__":
    main()
