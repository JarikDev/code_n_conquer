import csv
from docx import Document

# Load the Word document
doc = Document("data/search_examples.docx")

# Extract data from tables
data = []
for table in doc.tables:
    for row in table.rows:
        row_data = [cell.text.strip() for cell in row.cells]
        data.append(row_data)

# Write data to CSV
with open('data/search_examples.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(data)