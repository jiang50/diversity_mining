import csv

lines = []
with open('dic_v3_2.txt', 'r') as readFile:
    for line in readFile:
        if line[-1] == '\n':
            line = line[:-1]

        lines.append([line])


with open('dic2205.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
