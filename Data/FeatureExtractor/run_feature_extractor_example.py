from handle_chart import HandleChart

'''
Example: Get the <uid>.DF.json file for all tables of schema <uid>, in this case uid is '0'. Notice uid is a string of numbers.
requirement: <uid>.json and <uid>.t<tuid>.table.json in data_path.
'''
hc = HandleChart()
hc.ExtractForChart(data_path='./example/data/', output_path='./example/output/', uid='0')
