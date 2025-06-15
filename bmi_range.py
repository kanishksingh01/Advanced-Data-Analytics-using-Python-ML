Height = input("Height in inches: ")
Weight = input("Weight in pounds: ")

bmi = (int(Weight) / (int(Height) * int(Height))) * 703
weight_low = round(18.5 * (int(Height) * int(Height)) / 703)
weight_high = round(24.9 * (int(Height) * int(Height)) / 703)

print(f"For a height of {Height} inches, a healthy BMI is ranged between {weight_low} and {weight_high}.")