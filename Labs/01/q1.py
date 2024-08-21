# Calculate the body mass index (BMI) of two variables input by the user, where:
# BMI= weight/(height)^2.

weight = float(input("Enter your weight (in kg): "))
height = float(input("Enter your height (in meters): "))

BMI = weight/(height ** 2)

print("Your BMI is:", BMI)
