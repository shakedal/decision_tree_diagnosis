import xlsxwriter

def write_to_excel(all_results, file_name):
    # create excel file
    file_name = f"results/{file_name}.xlsx"
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    
    # write headers
    dict_example = all_results[0]
    index_col = {}
    col_num = 0
    for key in dict_example.keys():
        worksheet.write(0, col_num, key)
        index_col[key] = col_num
        col_num += 1
    
    # write values
    row_num = 1
    for dict_res in all_results:
        for key, value in dict_res.items():
            if type(value) in (list, set, dict):
                value = str(value)
            col_num = index_col[key]
            try:
                worksheet.write(row_num, col_num, value)
            except TypeError:
                print(f"{dict_res['dataset']} {dict_res['size']} - problem with key: '{key}', value: {value}")
        row_num += 1
    
    workbook.close()
    
if __name__ == '__main__':
    pass