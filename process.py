import csv
lines = []
with open('dic1156_1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if row[1] != '0':
            lines.append(row)

with open('dic_final.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
