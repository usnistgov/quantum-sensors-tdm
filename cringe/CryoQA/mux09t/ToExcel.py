import openpyxl as xl

wb = xl.Workbook()
activeBook = wb.active

activeBook["A1"] = "Chip ID"
activeBook["B1"] = "Row"
activeBook["C1"] = "Icmin (uA)"
activeBook["D1"] = "Icmax (uA)"

activeBook["A" + str((self.channel*self.rows) + 2)] = self.chipID
