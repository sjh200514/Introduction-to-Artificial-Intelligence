#导入z3库
from z3 import *

#输入待求解的数独,一共输入9行，每行9个数字（输入的数独中空位填入0），每行中相邻的两个数用空格隔开
sudoku_input = [[0 for _ in range(9)] for _ in range(9)]  
for i in range(9):
    line = input().strip()
    numbers = line.split()  
    for j in range(9):  
        num = int(numbers[j])
        sudoku_input[i][j] = num 

X = [ [ Int("x_%s_%s" % (i+1, j+1)) for j in range(9) ] for i in range(9) ]

#为每个单元格创建一个约束，确保单元格中的值在1到9之间
num_valid = [ And(1 <= X[i][j], X[i][j] <= 9) for i in range(9) for j in range(9) ]

#为每一行创建一个约束，确保每一行中的数字都是唯一的
rows_valid = [ Distinct(X[i]) for i in range(9) ]

#为每一列创建一个约束，确保每一列中的数字都是唯一的
cols_valid = [ Distinct([ X[i][j] for i in range(9) ]) for j in range(9) ]

#为每一个九宫格创建一个约束，确保每个九宫格中九个格子的数字都是唯一的。
squares_valid = [ Distinct([ X[3*dx + i][3*dy + j]for i in range(3) for j in range(3) ])
for dx in range(3) for dy in range(3) ]

#一个合法的数独需要同时满足上述约束
sudoku_valid = num_valid + rows_valid + cols_valid + squares_valid

#对输入的待求解的空位(sudoku_input[i][j] = 0的位置)不施加额外的约束
sudoku_solve_rule = [ If(sudoku_input[i][j] == 0,True,X[i][j] == sudoku_input[i][j])
for i in range(9) for j in range(9) ]

s = Solver()
s.add(sudoku_valid + sudoku_solve_rule)
if s.check() == sat:
    m = s.model()
    r = [ [ m.evaluate(X[i][j]) for j in range(9) ] for i in range(9) ]
    #输出9行，每行9个数，同一行中相邻两个数之间用空格隔开
    for line in r:
      print(' '.join(map(str, line)))
else:
    print ("failed to solve")