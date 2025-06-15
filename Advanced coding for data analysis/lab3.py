#!/usr/bin/env python
# coding: utf-8

# In[1]:


def route_time():
    num_legs = int(input("Enter the number of legs: "))
    total_time = 0.0

    for i in range(num_legs):
        distance = float(input(f"Enter the distance for leg {i+1} in miles: "))
        speed = float(input(f"Enter the speed for leg {i+1} in miles per hour: "))
        if speed <= 0:
            print("Speed must be greater than 0.")
            return
        leg_time = distance / speed
        total_time += leg_time

    total_time_minutes = round(total_time * 60, 2)
    print(f"Total time to finish {num_legs} legs is {total_time_minutes} minutes.")

# Example usage
if __name__ == "__main__":
    route_time()


# In[3]:


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


# In[ ]:




