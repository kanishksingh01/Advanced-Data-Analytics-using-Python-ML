def calcTuition(credits):
    return credits

def calcTuition(credits):
    if credits >= 12:
        return 20000
    elif credits >= 1:
        return 1200 + 1700 * credits
    else:
        return 0

def main():
    try:
        credits = int(input("Enter the number of credits: "))
        if credits < 0:
            print("Number of credits cannot be negative.")
        else:
            tuition = calcTuition(credits)
            print(f"The tuition for {credits} credits is ${tuition}.")
    except ValueError:
        print("Please enter a valid number of credits.")

if __name__ == "__main__":
    main()