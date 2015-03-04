# -*- coding: utf-8 -*-
import xlrd
import pandas as pd

def str2float(x):
    if isinstance(x, str):
        if x.strip() == '':
            return nan
        else:
            return float64(x)
    return x


def read_tecan_xlsx(filename, label, sheet=None, max_time=None, well2strain=None):
    wb = xlrd.open_workbook(filename)
    if sheet == None:
        for sh in wb.sheets():
            if sh.nrows > 0: 
                break
    else:
        sh = wb.sheets()[sheet]
        
    for i in range(sh.nrows):
        row = sh.row_values(i)
        if row[0] == label:
            break
            
    data = {}
    for i in range(i+1, sh.nrows):
        row = sh.row(i)
        if row[0].value == '':
            break
        data[row[0].value] = [x.value for x in row[1:] if x.ctype == 2]
    
    min_length = min(map(len, data.values()))
    for k,v in data.items():
        data[k] =  v[:min_length]
    
    df = pd.DataFrame(data)
    df = pd.melt(df, id_vars=('Time [s]',u'Temp. [°C]','Cycle Nr.'), var_name='Well', value_name=label)
    df.rename(columns={'Time [s]': 'Time'}, inplace=True)
    df.Time = df.Time / 3600.
    df['Letter'] = map(lambda x: x[0], df.Well)
    df['Number'] = map(lambda x: int(x[1:]), df.Well)
    if well2strain:
        df['Strain'] = map(well2strain, df.Well)
    if not max_time:
        max_time = df.Time.max()
    df = df[df.Time < max_time]    
    return df