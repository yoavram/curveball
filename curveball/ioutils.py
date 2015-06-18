#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoavram+github@gmail.com>
import xlrd
import numpy as np
import pandas as pd
from string import ascii_uppercase
from scipy.io import loadmat
from lxml import etree
import re
import datetime
import dateutil.parser
from glob import glob
import os.path


MAT_VERSION = '1.0'


def read_tecan_xlsx(filename, label, sheet=None, max_time=None, plate=None):
    """Reads growth measurements from a Tecan Infinity Excel output file.

    Args:
        - filename: Path to the file (:py:class:`str`).
        - label: String or sequence of strings of measurment names used as titles of the data tables in the file.
        - sheet: Sheet number, if known. Otherwise the function will try to guess the sheet.
        - max_time: The maximal time, in hours (:py:class:`int`).
        - plate: A pandas DataFrame object representing a plate, usually generated by reading a CSV file generated by `Plato <http://plato.yoavram.com/>`_.

    Returns:
        A :py:class:`pandasDataFrame` containing the columns:

        - `Time` (:py:class:`int`, in hours)
        - `Temp. [°C]` (:py:class:`float`)
        - `Cycle Nr.` (:py:class:`int`)
        - `Well` (:py:class:`str`): the well name, usually a letter for the row and a number of the column.
        - `Row` (:py:class:`str`): the letter corresponding to the well row.
        - `Col` (:py:class:`str`): the number corresponding to the well column.
        - `Strain` (:py:class:`str`): if a `plate` was given, this is the strain name corresponding to the well from the plate.
        - `Color` (:py:class:`str`, hex format): if a `plate` was given, this is the strain color corresponding to the well from the plate.

        There will also be a separate column for each label, and if there is more than one label, a separate `Time` and `Temp. [°C]` column for each label.

    Example:
    
    >>> plate = pd.read_csv("plate_templates/G-RG-R.csv")
    >>> df = curveball.ioutils.read_tecan_xlsx("data/yoavram/210115.xlsx", ('OD','Green','Red'), max_time=12, plate=plate)
    >>> df.shape
    (8544, 9)
    """
    wb = xlrd.open_workbook(filename)
    dateandtime = datetime.datetime.now() # default

    if isinstance(label, str):
        label = [label]

    label_dataframes = []
    for lbl in label:
        sheet_dataframes = []
        for sh in wb.sheets():
            if sh.nrows == 0:
                continue # to next sheet
        ## FOR sheet
            for i in range(sh.nrows):
                ## FOR row
                row = sh.row_values(i)

                if row[0].startswith('Date'):
                    if isinstance(row[1], str) or isinstance(row[1], unicode):                        
                        date = str.join('', row[1:])
                        next_row = sh.row_values(i+1)                
                        time = str.join('', next_row[1:])
                        dateandtime = dateutil.parser.parse("%s %s" % (date, time))
                    else:
                        print "Warning: date row contained non-string:", row[1], type(row[1])

                if row[0] == lbl:
                    break
                ## FOR row ENDS

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
            df.Time = map(lambda t: dateandtime + datetime.timedelta(0, t), df.Time)        
            df['Row'] = map(lambda x: x[0], df.Well)
            df['Col'] = map(lambda x: int(x[1:]), df.Well)
            sheet_dataframes.append(df)
    ## FOR sheet ENDS
        if len(sheet_dataframes) == 0:
            df = pd.DataFrame()
        elif len(sheet_dataframes) == 1:
            df = sheet_dataframes[0]
        else:
            df = pd.concat(sheet_dataframes)

        min_time = df.Time.min()
        df.Time = map(lambda t: (t - min_time).total_seconds()/3600., df.Time)
        if not max_time is None:
            df = df[df.Time <= max_time]
        label_dataframes.append((lbl,df))

    if len(label_dataframes) == 0: # dataframes
        return pd.DataFrame()
    if len(label_dataframes) == 1: # just one dataframe
        df = label_dataframes[0][1]
    else: # multiple dataframes, merge together
        # FIXME last label isn't used as a suffix, not sure why
        lbl,df = label_dataframes[0]
        lbl = '_' + lbl
        for lbli,dfi in label_dataframes[1:]:
            lbli = '_' + lbli
            df = pd.merge(df, dfi, on=('Cycle Nr.','Well','Row','Col'), suffixes=(lbl,lbli))
    if plate is None:
        df['Strain'] = 0
        df['Color'] = '#000000'
    else:
        df = pd.merge(df, plate, on=('Row','Col'))
    return df


def read_tecan_mat(filename, time_label='tps', value_label='plate_mat', value_name='OD', plate_width=12, max_time=None, plate=None):
    """Reads growth measurements from a Matlab mat file generated from the results of a Tecan Infinity plate reader.
    """
    mat = loadmat(filename, appendmat=True)
    if mat['__version__'] != MAT_VERSION:
        print "Warning: expected mat file version %s but got %s" % (MAT_VERSION, mat['__version__'])
    t = mat[time_label]
    t = t.reshape(max(t.shape))
    y = mat[value_label]
    assert y.shape[1] == t.shape[0]

    df = pd.DataFrame(y.T, columns=np.arange(y.shape[0]) + 1)
    df['Time'] = t / 3600.
    df['Cycle Nr.'] = np.arange(1, 1 + len(t))
    df = pd.melt(df, id_vars=('Cycle Nr.', 'Time'), var_name='Well', value_name=value_name)
    df['Well'] = map(lambda w: ascii_uppercase[(int(w)-1)/plate_width] + str(w%plate_width if w%plate_width > 0 else plate_width), df['Well'])
    df['Row'] = map(lambda w: w[0], df['Well'])
    df['Col'] = map(lambda w: int(w[1:]), df['Well'])
    
    if plate is None:
        df['Strain'] = 0
        df['Color'] = '#000000'
    else:
        df = pd.merge(df, plate, on=('Row','Col'))
    if not max_time:
        max_time = df.Time.max()
    df = df[df.Time < max_time]       
    return df


def read_tecan_xml(filename, label='OD', max_time=None, plate=None):
    """
    Reads growth measurements from a Tecan Infinity XML output file.

    This function was adapted from `choderalab/assaytools <https://github.com/choderalab/assaytools/blob/908471e7976e207df3f9b0e31b2a89f84da40607/AssayTools/platereader.py>`_ (licensed under LGPL).

    Args:
        - filename : :py:class:`str`, the name of the XML file to be read. Use * and ? in filename to read multiple files and parse them into a single data frame.
        - label: :py:class:`str`, measurment name used as `Name` in the measurement sections in the file.
        - max_time: The maximal time, in hours (:py:class:`int`).
        - plate: A pandas DataFrame object representing a plate, usually generated by reading a CSV file generated by `Plato <http://plato.yoavram.com/>`_.

    Returns:
        A :py:class:`pandasDataFrame` containing the columns:

        - `Time` (:py:class:`int`, in hours)
        - `Well` (:py:class:`str`): the well name, usually a letter for the row and a number of the column.
        - `Row` (:py:class:`str`): the letter corresponding to the well row.
        - `Col` (:py:class:`str`): the number corresponding to the well column.
        - `Filename` (:py:class:`str`): the filename from which this measurement was read.
        - `Strain` (:py:class:`str`): if a `plate` was given, this is the strain name corresponding to the well from the plate.
        - `Color` (:py:class:`str`, hex format): if a `plate` was given, this is the strain color corresponding to the well from the plate.

    There will also be a separate column for the value of the label.

    Example:

    >>> plate = pd.read_csv("plate_templates/checkerboard.csv")
    >>> df = curveball.ioutils.read_tecan_xlsx("data/dorith/*.xml", 'OD', plate=plate)
    >>> df.shape
    (2016, 8)
    """
    dataframes = []
    for filename in glob(filename):
        # Parse XML file into nodes.
        root_node = etree.parse(filename)

        # Build a dict of section nodes.
        section_nodes = { section_node.get('Name') : section_node for section_node in root_node.xpath("/*/Section") }

        # Process all sections.
        if label not in section_nodes:
            return pd.DataFrame()

        section_node = section_nodes[label]
        
        # Get the time of measurement
        time_start = section_node.attrib['Time_Start']

        # Get a list of all well nodes
        well_nodes = section_node.xpath("*/Well")
        
        # Process all wells into data.
        well_data = []
        for well_node in well_nodes:
            well = well_node.get('Pos')
            value = float(well_node.xpath("string()"))
            well_data.append({'Well': well, label: value})

        # Add to data frame
        df = pd.DataFrame(well_data)
        df['Row'] = map(lambda x: x[0], df.Well)
        df['Col'] = map(lambda x: int(x[1:]), df.Well)
        df['Time'] = dateutil.parser.parse(time_start)
        df['Filename'] = os.path.split(filename)[-1]
        dataframes.append(df)
    df = pd.concat(dataframes)
    min_time = df.Time.min()
    df.Time = map(lambda t: (t - min_time).total_seconds()/3600., df.Time)
    if plate is None:
        df['Strain'] = 0
        df['Color'] = '#000000'
    else:
        df = pd.merge(df, plate, on=('Row','Col'))
    if not max_time is None:
        df = df[df.Time <= max_time]
    return df


def read_sunrise_xlsx(filename, label='OD', max_time=None, plate=None):
    dataframes = []
    files = glob(filename)
    if not files:
        return pd.DataFrame()
    for filename in files:
        wb = xlrd.open_workbook(filename)
        for sh in wb.sheets():
            if sh.nrows > 0:
                break
        parse_data = False # start with metadata
        index = []
        data = []

        for i in range(sh.nrows):
            row = sh.row_values(i)
            if row[0] == 'Date:':
                date = filter(lambda x: isinstance(x,float), row[1:])[0]
                date = xlrd.xldate_as_tuple(date, 0)        
            elif row[0] == 'Time:':
                time = filter(lambda x: isinstance(x,float), row[1:])[0]
                time = xlrd.xldate_as_tuple(time, 0)
            elif row[0] == '<>':
                columns = map(int,row[1:])
                parse_data = True
            elif row[0] == '' and parse_data:
                break
            elif parse_data:
                index.append(row[0])
                data.append(map(float, row[1:]))
                
        dateandtime = date[:3] + time[-3:]
        dateandtime = datetime.datetime(*dateandtime)
        
        df = pd.DataFrame(data, columns=columns, index=index)
        df['Row'] = index
        df = pd.melt(df, id_vars='Row', var_name='Col', value_name=label)
        df['Time'] = dateandtime
        df['Well'] = map(lambda x: x[0] + str(x[1]), zip(df.Row,df.Col))
        df['Filename'] = os.path.split(filename)[-1]
        dataframes.append(df)
    df = pd.concat(dataframes)
    min_time = df.Time.min()
    df.Time = map(lambda t: (t - min_time).total_seconds()/3600., df.Time)
    if plate is None:
        df['Strain'] = 0
        df['Color'] = '#000000'
    else:
        df = pd.merge(df, plate, on=('Row','Col'))
    if not max_time is None:
        df = df[df.Time <= max_time]
    return df