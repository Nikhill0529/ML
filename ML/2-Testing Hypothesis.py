2.1 FIND-S algorithm (read training data from file, generate final specific hypothesis)

!pip install openpyxl 
import csv
import openpyxl
# Load the Excel workbook
workbook = openpyxl.load_workbook('/content/enjoysport.xlsx') # Correct path and also check file name
# Select the first sheet
sheet = workbook.active
# Create a temporary CSV file
with open('temp.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    # Iterate over the rows and write to the CSV file 
    for row in sheet.iter_rows():
        writer.writerow([cell.value for cell in row]) 
num_attributes = 6
a = []
print("\n The Given Training Data Set \n")
# Now, you can read the data using csv.reader

#change 2
"""
with open('temp.csv', 'r', encoding='utf-8') as csvFile: # Use the temporary CSV file
    reader = csv.reader(csvFile)
    for row in reader:
        a.append(row) 
        print(row)
"""
#write below code
with open('temp.csv', 'r', encoding='utf-8') as csvFile: # Use the temporary CSV file
    reader = csv.reader(csvFile)
    next(reader)  # Skip the header row
    for row in reader:
        a.append(row) 
        print(row)

print("\n The initial value of hypothesis: ") 
hypothesis = ['0'] * num_attributes 
print(hypothesis)

#change 2
#for j in range(0, num_attributes):
#    hypothesis[j] = a[0][j]
#write below code
for row in a:
    if row[num_attributes].strip().lower()=='yes':
        hypothesis=row[:num_attributes]
        break

print("\n Find S: Finding a Maximally Specific Hypothesis\n")
for i in range(0, len(a)):
    if a[i][num_attributes] == 'yes':
        for j in range(0, num_attributes):
            if a[i][j] != hypothesis[j]:
                hypothesis[j] = '?'
            else:
                hypothesis[j] = a[i][j]
            print(" For Training instance No:{0} the hypothesis is ".format(i), hypothesis) 
print("\n The Maximally Specific Hypothesis for a given Training Examples :\n") 
print(hypothesis)














rough



#Step 1:- Import packages
import csv
import openpyxl

#Step 2:- Load the Excel workbook
workbook = openpyxl.load_workbook('/content/enjoysport.xlsx')

#Step 3:- Select the first sheet
sheet = workbook.active

#Step 4:- Create a temporary CSV file and write the da
with open('temp.csv','w',newline='',encoding='utf-8') as csv_file:
  writer = csv.writer(csv_file)
  #Iterate over the rows and write to the CSV file
  for row in sheet.iter_rows():
    writer.writerow([cell.value for cell in row])
num_attributes=6
a=[]
print("\n The Given Training Data Set \n")
# Now, you can read the data using csv.reader
with open('temp.csv', 'r', encoding='utf-8') as csvFile:  # Use the temporary CSV file
  reader = csv.reader(csvFile)
  for row in reader:
    a.append(row)
    print(row)
print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)
for j in range(0, num_attributes):
  hypothesis[j] = a[0][j]
print("\n Find S: Finding a Maximally Specific Hypothesis\n")
for i in range(0, len(a)):
  if a[i][num_attributes] == 'yes':
    for j in range(0, num_attributes):
      if a[i][j] != hypothesis[j]:
        hypothesis[j] = '?'
      else:
        hypothesis[j] = a[i][j]
      print(" For Training instance No:{0} the hypothesis is ".format(i), hypothesis)
print("\n The Maximally Specific Hypothesis for a given Training Examples :\n")
print(hypothesis)


