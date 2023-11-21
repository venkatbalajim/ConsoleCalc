# Import the necessary modules.
from functools import reduce
import math
import numpy as np

# Title of the program. 
print("\n\nWelcome to the CLI Calculator :)\n\n")

# Dictionary of the calculations and their IDs available in the program. 
calculation_names = {
    1: "Addition", 2: "Subtraction", 3: "Multiplication", 4: "Division",
    5: "Remainder", 6: "Exponential", 7: "Reciprocal", 8: "Percentage",
    9: "Rounded Value", 10: "Floor and Ceil Values", 11: "LCM and GCD",
    12: "Factorial", 13: "Combinations", 14: "Permutations", 15: "Pascal Triangle",
    16: "Sine", 17: "Cosine", 18: "Tangent", 19: "Cosecant", 20: "Secant", 21: "Cotangent",
    22: "Arcsine", 23: "Arccosine", 24: "Arctangent", 25: "Arccosecant", 26: "Arcsecant", 27: "Arccotangent",
    28: "User Defined Base", 29: "Natural Base",
    30: "Matrix Addition", 31: "Matrix Subtraction", 32: "Matrix Multiplication",
    33: "Scalar Multiplication", 34: "Matrix Exponential", 35: "Determinant",
    36: "Trace", 37: "Inverse", 38: "Transpose", 39: "Rank", 40: "Adjoint of a Matrix"
}

# List of all the categories of calculations. 
categories = {
    "Fundamental Calculations": [
        "Addition", "Subtraction", "Multiplication", "Division",
        "Remainder", "Exponential", "Reciprocal", "Percentage",
        "Rounded Value", "Floor and Ceil Values", "LCM and GCD"
    ],
    "Combinatoric Calculations": [
        "Factorial", "Combinations", "Permutations", "Pascal Triangle"
    ],
    "Trigonometric Calculations": [
        "Sine", "Cosine", "Tangent", "Cosecant", "Secant", "Cotangent"
    ],
    "Inverse Trigonometric Calculations": [
        "Arcsine", "Arccosine", "Arctangent", "Arccosecant", "Arcsecant", "Arccotangent"
    ],
    "Logarithmic Calculations": [
        "User Defined Base", "Natural Base"
    ],
    "Matrix Calculations": [
        "Matrix Addition", "Matrix Subtraction", "Matrix Multiplication",
        "Scalar Multiplication", "Matrix Exponential", "Determinant",
        "Trace", "Inverse", "Transpose", "Rank", "Adjoint of a Matrix"
    ]
}

# Addition calculation.
def addition():
    add_numbers = list(map(int, input("Enter all the numbers separately : ").split()))
    print("Performing addition calculation ...")
    add_answer = sum(add_numbers)
    return "The result is {}".format(add_answer)

# Subtraction calculation. 
def subtraction():
    subtract_numbers = list(map(int, input("Enter two numbers to perform subtraction : ").split()))
    print("Performing subtraction calculation ...")
    subtract_result = int(subtract_numbers[0]) - int(subtract_numbers[1])
    return "The result is {}".format(subtract_result)

# Multiplication calculation. 
def multiplication():
    multiply_numbers = list(map(int, input("Enter all the numbers separately : ").split()))
    print("Performing multiplication calculation ...")
    multiply_result = reduce((lambda x, y: x * y), multiply_numbers)
    return "The result is {}".format(multiply_result)

# Division calculation. 
def division():
    divide_numbers = list(map(float, input("Enter two numbers to perform division : ").split()))
    if float(divide_numbers[1]) == 0:
        return "Cannot divide by zero."
    else:
        print("Performing division calculation ...")
        divide_result = float(divide_numbers[0]) / float(divide_numbers[1])
        return "The result is {}".format(divide_result)

# Remainder/Modulus calculation. 
def remainder():
    rem_numbers = list(map(int, input("Enter two numbers for modulus/remainder operation : ").split()))
    if int(rem_numbers[1]) == 0:
        return "Cannot find a remainder of division by zero."
    else:
        print("Performing modulus/remainder calculation ...")
        rem_result = int(rem_numbers[0]) % int(rem_numbers[1])
        return "The result is {}".format(rem_result)

# Exponential calculation. 
def exponential():
    exp_base = int(input("Enter base number : "))
    exp_power = float(input("Enter power value : "))
    print("Performing exponentiation calculation ...")
    if exp_power >= 0:
        exp_result = pow(exp_base, exp_power)
    else:
        exp_result = exp_base ** (1 / abs(exp_power))
    return "The result is {}".format(exp_result)

# Reciprocal calculation. 
def reciprocal():
    recp_number = float(input("Enter a positive number : "))
    print("Performing reciprocal calculation ...")
    if recp_number == 0:
        return "The result is Infinity"
    elif recp_number <0:
        recp_result = 1.0 / float(abs(recp_number))
        return "Reciprocal is {}".format(-recp_result)
    else:
        recp_result = 1.0 / float(recp_number)
        return "Reciprocal is {}".format(recp_result)

# Percentage calculation. 
def percentage():
    print("1. Percentage of particular portion\n2. Portion value from the percentage")
    percent_choice = int(input("Enter the choice (1/2) : "))
    
    if percent_choice == 1:
        part_values = list(map(int, input("Enter the total amount and specific portion value separately : ").split()))
        print("Calculating the percentage ...")
        perc_portion = (part_values[1] / part_values[0]) * 100
        return f"The percentage of portion is {perc_portion}"
    elif percent_choice == 2:
        total_amount, percent_of_portion = map(int, input("Enter the total amount and percentage of portion separately : ").split())
        print("Calculating the portion(s) ...")
        portion_value = (percent_of_portion * total_amount) / 100
        return f"The value of portion is {portion_value}"
    else:
        return "Invalid Choice!"

# Rounding value calculation. 
def rounded_value():
    round_nums = list(map(float, input("Enter all decimal numbers separately : ").split()))
    decimal_place = int(input("Enter how many decimal places you want in the entered numbers : "))
    print("Rounding the value(s) ...")
    rnd_results = []
    for num in round_nums:
        rnd_results.append(round(num, decimal_place))
    return "Rounded values are : {}".format(rnd_results)

# Celing and Flooring values calculation.
def floor_and_ceil_values():
    ceil_floor_numbers = list(map(float, input("Enter all numbers to be floored or celled : ").split()))
    print("Performing the ceil and floor calculations ...")
    ceil_numbers = []
    floor_numbers = []
    for num in ceil_floor_numbers:
        ceil_num = math.ceil(num)
        floor_num = math.floor(num)
        ceil_numbers.append(ceil_num)
        floor_numbers.append(floor_num)
    print("Floors and Ceils results are as follows : ")
    print("Ceiling values of given numbers are : ", ceil_numbers)
    print("Floor values of given numbers are : ", floor_numbers)
    return "Ceil and Floor calculation done successfully."

# LCM and GCD calculations. 
def lcm_and_gcd():
    print("Choose an option:\n1. Calculate LCM.\n2. Calculate GCD.")
    calc_option = int(input("Enter your choice (1/2): "))
    if calc_option == 1:
        lcm_list = list(map(int, input("Enter two numbers separated by space : ").split()))
        print("Calculating LCM...")
        lcm = math.lcm(max(lcm_list), min(lcm_list))
        return f"LCM of {lcm_list} is {lcm}"
    elif calc_option == 2:
        gcd_list = list(map(int, input("Enter two numbers separated by space : ").split()))
        print("Calculating GCD...")
        gcd = math.gcd(max(gcd_list), min(gcd_list))
        return f"GCD of {gcd_list} is {gcd}"
    else:
        return "Invalid Choice!"

# Factorial calculation.
def factorial():
    fact_number = int(input("Enter a number to find its factorial : "))
    print("Calculating factorial...")
    result = math.factorial(fact_number)
    return f"The factorial of {fact_number} is {result}"

# Combination calculation. 
def combinations():
    combi_numbers = list(map(int, input("Enter two numbers separated by space : ").split()))
    n = max(combi_numbers)
    k = min(combi_numbers)
    print("Calculating combination...")
    result = math.comb(n, k)
    return f"Combinations of {n} taken {k} at a time is {result}"

# Permutation calculation.
def permutations():
    permute_numbers = list(map(int, input("Enter two numbers separated by space : ").split()))
    n = max(permute_numbers)
    r = min(permute_numbers)
    print("Calculating permutations with repetition...")
    result = math.factorial(n + r - 1) // (math.factorial(r) * math.factorial(n - 1))
    return f"Permutations of {n} with repetition {r} is {result}"

# Pascal triangle calculation. 
def pascal_triangle():
    rows = int(input("Enter the number of rows for Pascal Triangle : "))
    print("\nPascal's Triangle : \n")
    for i in range(rows):
        for j in range(rows - i):
            print(" ", end=" ")
        for j in range(i + 1):
            coef = math.comb(i, j)
            print(coef, end=" ")
        print()
    return "\nEnd of Pascal Triangle."

# Trigonometric and Inverse trigonometric calculations. 
def sine():
    angle = float(input("Enter an angle in degrees : "))
    radian = math.radians(angle)
    sin_value = math.sin(radian)
    return f"Sin({angle}) = {sin_value}"

def cosine():
    angle = float(input("Enter an angle in degrees : "))
    radian = math.radians(angle)
    cos_value = math.cos(radian)
    return f"Cos({angle}) = {cos_value}"

def tangent():
    angle = float(input("Enter an angle in degrees : "))
    radian = math.radians(angle)
    tan_value = math.tan(radian)
    return f"Tan({angle}) = {tan_value}"

def cosecant():
    angle = float(input("Enter an angle in degrees : "))
    radian = math.radians(angle)
    cosec_value = 1 / math.sin(radian)
    return f"Cosec({angle}) = {cosec_value}"

def secant():
    angle = float(input("Enter an angle in degrees : "))
    radian = math.radians(angle)
    sec_value = 1 / math.cos(radian)
    return f"Sec({angle}) = {sec_value}"

def cotangent():
    angle = float(input("Enter an angle in degrees : "))
    radian = math.radians(angle)
    cot_value = 1 / math.tan(radian)
    return f"Cot({angle}) = {cot_value}"

def arcsine():
    x = float(input("Enter a value between -1 and 1: "))
    asin_value = math.degrees(math.arcsin(x))
    return f"Arcsin({x}) = {asin_value} degree"

def arccosine():
    x = float(input("Enter a value between -1 and 1: "))
    acos_value = math.degrees(math.acos(x))
    return f"Arccos({x}) = {acos_value} degree"

def arctangent():
    y = float(input("Enter the Y coordinate: "))
    x = float(input("Enter the X coordinate: "))
    atan_value = math.degrees(math.atan2(y, x))
    return f"Arctan({y}, {x}) = {atan_value} degree"

def arccosecant():
    x = float(input("Enter a value between -1 and 1: "))
    acsc_value = math.degrees(math.acos(1/x))
    return f"Arccsc({x}) = {acsc_value} degree"

def arcsecant():
    x = float(input("Enter a value between -1 and 1: "))
    asec_value = math.degrees(math.acos(x/(abs(x))))
    return f"Arcsec({x}) = {asec_value} degree"

def arccotangent():
    x = float(input("Enter a value between -1 and 1: "))
    acot_value = math.degrees(math.atan(1/x))
    return f"Arccot({x}) = {acot_value} degree"

# Logarithm with user defined base calculation. 
def user_defined_base():
    num = float(input('Please enter a positive real number: '))
    base = int(input('Enter the base of logarithm (must be an integer greater than or equal to 2): '))
    if num <= 0 or base < 2:
        return "Invalid input"
    else:
        result = math.log(num, base)
        return f'The logarithm of {num} to the base {base} is approximately {result}'

# Natural logarithm calculation.
def natural_base():
    num = float(input("Please enter a positive real number: "))
    if num <= 0:
        return "Invalid input"
    else:
        result = math.log(num)
        return f'The natural logarithm of {num} is approximately {result}'

# Matrix addition calculation.
def matrix_addition():
    rows = int(input("Enter the number of rows for the matrices: "))
    cols = int(input("Enter the number of columns for the matrices: "))
    matrix1 = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix 1: ")) for j in range(cols)] for i in range(rows)]
    matrix2 = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix 2: ")) for j in range(cols)] for i in range(rows)]
    sum_matrix = [[matrix1[i][j] + matrix2[i][j] for j in range(cols)] for i in range(rows)]
    for row in sum_matrix:
        for elt in row:
            print(elt, end=" ")
        print()
    return "Matrix addition done successfully."

# Matrix subtraction calculation. 
def matrix_subtraction():
    rows = int(input("Enter the number of rows for the matrices: "))
    cols = int(input("Enter the number of columns for the matrices: "))
    matrix1 = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix 1: ")) for j in range(cols)] for i in range(rows)]
    matrix2 = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix 2: ")) for j in range(cols)] for i in range(rows)]
    diff_matrix = [[matrix1[i][j] - matrix2[i][j] for j in range(cols)] for i in range(rows)]
    for row in diff_matrix:
        for elt in row:
            print(elt, end=" ")
        print()
    return "Matrix subtraction calculation done successfully."

# Scalar multiplication calculation. 
def scalar_multiplication():
    rows = int(input("Enter the number of rows for the matrices: "))
    cols = int(input("Enter the number of columns for the matrices: "))
    scalar = float(input("Enter the scalar value: "))
    matrix1 = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix 1: ")) for j in range(cols)] for i in range(rows)]
    mult_matrix = [[scalar * matrix1[i][j] for j in range(cols)] for i in range(rows)]
    for row in mult_matrix:
        for elt in row:
            print(elt, end=" ")
        print()
    return "Scalar multiplication done successfully."

# Determinant calculation. 
def determinant():
    rows = int(input("Enter the number of rows for the matrix: "))
    cols = int(input("Enter the number of columns for the matrix: "))
    matrix = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix: ")) for j in range(cols)] for i in range(rows)]
    matrix_np = np.array(matrix)
    det = round(np.linalg.det(matrix_np), 2)
    return f"The determinant of the matrix is {det}"

# Trace calculation. 
def trace():
    rows = int(input("Enter the number of rows for the matrix: "))
    cols = int(input("Enter the number of columns for the matrix: "))
    matrix = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix: ")) for j in range(cols)] for i in range(rows)]
    trace_sum = sum(matrix[i][i] for i in range(min(rows, cols)))
    return f"The trace of the matrix is {trace_sum}"

# Matrix exponential calculation. 
def matrix_exponential():
    rows = int(input("Enter the number of rows for the matrix: "))
    cols = int(input("Enter the number of columns for the matrix: "))
    matrix = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix: ")) for j in range(cols)] for i in range(rows)]
    matrix_np = np.array(matrix)
    exp_matrix = np.linalg.expm(matrix_np)
    for row in exp_matrix:
        for elt in row:
            print(elt, end = " ")
        print()
    return "Exponential operation of a matrix done successfully"

# Matrix multiplication calculation. 
def matrix_multiplication():
    rowsA = int(input("Enter the number of rows for first matrix: "))
    colsA = int(input("Enter the number of columns for first matrix: "))
    rowsB = int(input("Enter the number of rows for second matrix: "))
    colsB = int(input("Enter the number of columns for second matrix: "))
    if colsA != rowsB:
        return "These two matrices cannot be multiplied!"
    else:
        matrixA = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix A: ")) for j in range(colsA)] for i in range(rowsA)]
        matrixB = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix B: ")) for j in range(colsB)] for i in range(rowsB)]
        result = [[0 for _ in range(colsB)] for _ in range(rowsA)]
        for i in range(rowsA):
            for j in range(colsB):
                for k in range(colsA):
                    result[i][j] += matrixA[i][k] * matrixB[k][j]
        print("Resultant Matrix (A x B):")
        for row in result:
            print(' '.join(map(str, row)))
        return "Matrix multiplication done successfully"

# Inverse of a matrix calculation. 
def inverse():
    rows = int(input("Enter the number of rows for the matrix: "))
    cols = int(input("Enter the number of columns for the matrix: "))
    matrix = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix: ")) for j in range(cols)] for i in range(rows)]
    matrix_np = np.array(matrix)
    try:
        inv_mat = np.linalg.inv(matrix_np)
        for row in inv_mat:
            for elt in row:
                print(elt, end=" ")
            print()
        return "Inverse operation done successfully"
    except Exception as e:
        return f"An error occured. The error is explained below.\n{e}"

# Transpose of a matrix calculation. 
def transpose():
    rows = int(input("Enter the number of rows for the matrix: "))
    cols = int(input("Enter the number of columns for the matrix: "))
    matrix = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix: ")) for j in range(cols)] for i in range(rows)]
    rows = len(matrix)
    cols = len(matrix[0])
    transposed_matrix = [[matrix[j][i] for j in range(rows)] for i in range(cols)]
    for row in transposed_matrix:
        for elt in row:
            print(elt, end=" ")
        print()
    return "Transpose operation done successfully."

# Rank of a matrix calculation. 
def rank():
    rows = int(input("Enter the number of rows for the matrix: "))
    cols = int(input("Enter the number of columns for the matrix: "))
    matrix = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix: ")) for j in range(cols)] for i in range(rows)]
    matrix_np = np.array(matrix)
    rank_val = np.linalg.matrix_rank(matrix_np)
    return f"Rank of the given matrix is {rank_val}."

# Adjoint matrix calculation. 
def sub_matrix(matrix, i, j):
    return [row[:j] + row[j+1:] for row in (matrix[:i]+matrix[i+1:])]
def determinant_adjoint(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    else:
        det = 0
        for c in range(len(matrix)):
            sign = (-1) ** c
            sub_det = determinant_adjoint(sub_matrix(matrix, 0, c))
            det += sign * matrix[0][c] * sub_det
        return det
def adjoint():
    rows = int(input("Enter the number of rows for the matrix: "))
    cols = int(input("Enter the number of columns for the matrix: "))
    matrix = [[int(input(f"Enter element at position ({i+1}, {j+1}) for Matrix: ")) for j in range(cols)] for i in range(rows)]
    adjugate_matrix = np.zeros((len(matrix), len(matrix)), dtype=int)
    for p in range(len(matrix)):
        for q in range(len(matrix)):
            sign = (-1) ** (p + q)
            sub_det = determinant_adjoint(sub_matrix(matrix, q, p))
            cofactor = sub_det * sign
            adjugate_matrix[p][q] = cofactor
    for row in adjugate_matrix:
        for elt in row:
            print(elt, end = " ")
        print()
    return "Adjoint of a matrix done successfully"

# Switches to redirect to the selected calculation. 
switch_cases = {
    1: addition,
    2: subtraction,
    3: multiplication,
    4: division,
    5: remainder,
    6: exponential,
    7: reciprocal,
    8: percentage,
    9: rounded_value,
    10: floor_and_ceil_values,
    11: lcm_and_gcd,
    12: factorial,
    13: combinations,
    14: permutations,
    15: pascal_triangle,
    16: sine,
    17: cosine,
    18: tangent,
    19: cosecant,
    20: secant,
    21: cotangent,
    22: arcsine,
    23: arccosine,
    24: arctangent,
    25: arccosecant,
    26: arcsecant,
    27: arccotangent,
    28: user_defined_base,
    29: natural_base,
    30: matrix_addition,
    31: matrix_subtraction,
    32: matrix_multiplication,
    33: scalar_multiplication,
    34: matrix_exponential,
    35: determinant,
    36: trace,
    37: inverse,
    38: transpose,
    39: rank,
    40: adjoint
}

# Function to fetch the name of the calculation using the ID. 
def get_calculation_id(calculation_name):
    for key, value in calculation_names.items():
        if value == calculation_name:
            return key
    return "Invalid Calculation"

# Function to display all the calculations in the console. 
def display_calculations(category, calculations):
    print(f"\nCategory - {category} :\n")
    for idx, calc in enumerate(calculations, start=1):
        calc_id = get_calculation_id(calc)
        print(f"{calc_id}. {calc}")

# Function to redirect to the selected operation suing switches. 
def execute_case(selected_case):
    selected_function = switch_cases.get(selected_case)
    return selected_function()

# Print all the calculation names. 
print("All Available Calculations :\n")
for category, calculations in categories.items():
    display_calculations(category, calculations)

# Flag to check whether the user wants to repeat the execution of program. 
run = True
while run:

    # Getting the ID from the user. 
    print("\nKindly refer to the above calculations and their IDs.")
    choice = int(input("Enter the calculation number (1-50): "))

    # Showing the ID and its calculation name. 
    print("\nEntered choice :", choice)
    selected_calculation = calculation_names.get(choice, "Invalid ID")

    # If the entered ID is NOT in the dictionary ... 
    if selected_calculation == "Invalid ID":
        print("Please enter a valid ID for any of the calculations listed above.")
        print("")
        run = input("Do you want to continue? default is YES. (yes/no) ").lower()
        run = run == 'yes' if run in ['yes', 'no'] else True

    # If the entered ID is in the dictionary ...
    else:
        print("Selected Calculation :", selected_calculation)
        result = execute_case(choice)
        print(result)
        print("")
        run = input("Do you want to continue? default is YES. (yes/no) ").lower()
        run = run == 'yes' if run in ['yes', 'no'] else True
