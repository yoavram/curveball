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


def read_tecan_xlsx(filename, label, sheet=None, max_time=None, plate=None):
    """
    Reads a Tecan Infinity Excel file and returns a Pandas dataframe.
    """
    wb = xlrd.open_workbook(filename)
    if sheet == None:
        for sh in wb.sheets():
            if sh.nrows > 0:
                break
    else:
        sh = wb.sheets()[sheet]

    if isinstance(label, str):
        label = [label]

    dataframes = []
    for lbl in label:
        for i in range(sh.nrows):
            row = sh.row_values(i)
            if row[0] == lbl:
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
        df = pd.melt(df, id_vars=('Time [s]',u'Temp. [°C]','Cycle Nr.'), var_name='Well', value_name=lbl)
        df.rename(columns={'Time [s]': 'Time'}, inplace=True)
        df.Time = df.Time / 3600.
        df['Row'] = map(lambda x: x[0], df.Well)
        df['Col'] = map(lambda x: int(x[1:]), df.Well)
        if plate is None:
            df['Strain'] = 0
            df['Color'] = '#000000'
        else:
            df = pd.merge(df, plate, on=('Row','Col'))
        if not max_time:
            max_time = df.Time.max()
        df = df[df.Time < max_time]
        dataframes.append((lbl,df))

    if len(dataframes) == 0:
        return pd.DataFrame()
    if len(dataframes) == 1:
        return dataframes[0]
    else:
        # FIXME last label isn't used as a suffix, not sure why
        lbl,df = dataframes[0]
        lbl = '_' + lbl
        for lbli,dfi in dataframes[1:]:
            lbli = '_' + lbli
            df = pd.merge(df, dfi, on=('Cycle Nr.','Well','Row','Col','Strain','Color'), suffixes=(lbl,lbli))
        return df
