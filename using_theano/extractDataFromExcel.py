__author__ = 'prm14'
import xlrd

data = xlrd.open_workbook('data.xlsx')

table = data.sheet_by_name(u'Sheet1')
print(map(lambda x:x.encode('gbk'),table.col_values(2)))

