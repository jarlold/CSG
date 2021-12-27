import requests
import bs4
from bs4 import BeautifulSoup as bs
import re
import json
import pandas as pd
import numpy as np
from datetime import date
import copy


# Convert to number of days since year X
def convert_timestamp(timestamp, since=date(2000, 1, 1)):
    d = date(* [int(i2) for i2 in timestamp.split("-")])
    return (d - since).days


def yoink_financial_ratios(stock_ticker, stock_name):
    r = requests.get('https://www.macrotrends.net/stocks/charts/{}/{}/financial-ratios?freq=Q'.format(stock_ticker, stock_name))
    p = re.compile(r' var originalData = (.*?);\r\n\r\n\r',re.DOTALL)
    data = json.loads(p.findall(r.text)[0])
    headers = list(data[0].keys())
    headers.remove('popup_icon')
    headers.remove('field_name')
    result = []

    for row in data:
        soup = bs(row['field_name'], 'lxml')
        field_name = soup.select_one('a, span').text
        fields = list(row.values())
        fields.insert(0, str(field_name))
        result.append(fields)

    # remove some excess strings that come along for some reason
    for i in result: # Pass by reference error?
        for k in i:
            if k.__class__ is str and ("<" in k or ">" in k):
                i.remove(k)
            elif k.__class__ is str and "EBITDA Margin" in k:
                result.remove(i)

    # Remove the Pre-tax profit margin, for some reason this string does not evaluate as "pre tax profit margin"
    # I expect encoding fuckery, but this is an easier solution
    result.pop(6)
    names = []

    for i in result:
        names.append(i.pop(0))

    # Make transpose
    transpose = np.array(result).transpose().tolist()

    headers = [convert_timestamp(headers[i]) for i, _ in enumerate(headers)]

    # sort the list by time
    sorted_headers = sorted(headers, key=lambda x: x)

    for i3, _ in enumerate(transpose):
        start_date = headers[i3]
        start_date_index = sorted_headers.index(headers[i3])
        if not start_date_index+1 > len(sorted_headers)-1:
            end_date = sorted_headers[start_date_index+1]
        else:
            # assume the final date is good for 1 more quarter/year
            # this won't fuck up the scaling (unlike putting a big number at the end)
            end_date = headers[i3] - sorted_headers[start_date_index - 1] + headers[i3]
        transpose[i3].append(start_date)
        transpose[i3].append(end_date)

    # sort the list by time
    transpose = sorted(transpose, key=lambda x: x[-2])

    # if there are missing values, leave them as their old values
    for i, _ in enumerate(transpose):
        for k, _ in enumerate(transpose[i]):
            if transpose[i][k] == '':
                transpose[i][k] = transpose[i-1][k]

    # Convert list to floats
    transpose = np.array(transpose, dtype=np.float32).tolist()

    # do the feed forward stuff
    feed_forward = []
    for i, _ in enumerate(transpose):
        if i == 0: continue
        for k in range(2, int(transpose[i][-1] - transpose[i-1][-1])):
            a = copy.copy(transpose[i])
            a[-1] += k-2
            feed_forward.append(a)

    names.append("Start Date")
    names.append("End Date")
    df = pd.DataFrame(transpose, columns=names)
    return df


def fill_in_feed_forward(ogd, ffd):
    # Runs in len(ogd)*len(ffd) speed, which is poopy but oh well
    ffd_no_col = ffd.drop(columns=["Start Date", "End Date"])
    df_to_append = []
    for current_date in ogd["Date"]:
        for i, (start_date, end_date) in enumerate(zip(ffd["Start Date"], ffd["End Date"])):
            if current_date >= start_date and current_date < end_date:
                df_to_append.append(ffd_no_col.iloc[i].values.tolist())
            elif current_date > start_date and current_date == end_date:
                df_to_append.append(ffd_no_col.iloc[i].values.tolist())

    df_to_append = pd.DataFrame(df_to_append, columns=ffd_no_col.columns)
    return ogd.join(df_to_append)




