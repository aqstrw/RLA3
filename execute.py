''' This file executes the code for all three tasks '''

''' The code expects the data file to be in the workind directory '''

''' The files for each individual task are named
'task_1.py', 'task_2.py' and 'task_3.py' respectively '''


with open('task_1.py', "rb") as source_file:
    code = compile(source_file.read(), 'task_1.py', "exec")
exec(code)

with open('task_2.py', "rb") as source_file:
    code = compile(source_file.read(), 'task_2.py', "exec")
exec(code)

with open('task_3.py', "rb") as source_file:
    code = compile(source_file.read(), 'task_3.py', "exec")
exec(code)