#!/opt/homebrew/bin/python3

# author: Damián Sova
# date: 28.1.2023
# script which processes the text file with income information
# structure:
# Year: {
#     month:[
#         [
#             "XY czk" (czk),
#             "XY eur" (eur),
#             "haircuts sum XY" (eur)
#         ],
#         "XY + YZ + ..." (individual workdays income from haircuts),
#         "customers count: XY"
#     ]
# }

import csv
import os
import re
import argparse
import logging
import datetime
import calendar
from collections import defaultdict
from enum import StrEnum
from forex_python.converter import CurrencyRates
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logging.info(f"{f.__name__} took: {te - ts:2.2f} sec")
        return result
    return wrap

class Columns(StrEnum):
    """Class representing final columns of the exported csv file."""

    MONTH = "Month",
    WAGE_CZK = "WAGE(CZK)",
    WAGE_EUR = "WAGE(EUR)",
    HAIRCUT_EUR = "Haircut income (EUR)",
    INCOME_SUM = "Incom sum (EUR)"
    HAIRCUT_DAYS = "Haircut days",
    NUMBER_OF_CUSTOMERS = "Number of customers"
    MAX_HAIRCUT_DAY_INCOME = "Max haircut day income (EUR)"

def get_month_index(month, string = False):
    """Helper function to obtain convert rates for a given currency conversion."""

    return str(list(calendar.month_name).index(month)).zfill(2) if string else list(calendar.month_name).index(month)

def get_convert_rate(src_currency, dst_currency, month, year):
    """Helper function to obtain convert rates for a given currency conversion."""

    try:
        c = CurrencyRates()
        if month and year:
            month_num = get_month_index(month)
            dt = datetime.datetime(year, 1 if month_num == 12 else month_num + 1, 13, 14, 36, 28, 151012)
            return c.get_rate(src_currency, dst_currency, dt)
        else:
            return c.get_rate(src_currency, dst_currency)
    except:
        return 0.042

def get_column_index(column):
    """Returns the appropriate column index."""

    for idx, col in enumerate(Columns):
        if col == column:
            return idx


def read_file(input_file):
    """Function which reads and returns the content of a given file."""

    # Check if input file exists
    if not os.path.isfile(input_file):
        raise ValueError("Input file does not exist.")

    # Open the file for reading
    with open(input_file, 'r') as file:
        # Read the contents of the file
        content = file.read()

    if not content:
        raise ValueError("No content has been found in the input file.")
    
    return content


def export_year_csv(year, year_stats, stats_folder, output_file):
    """Creates a year statistics summary csv file from a given csv file."""

    # Check if exist the path
    if not os.path.exists(stats_folder):
        os.makedirs(stats_folder)
    if not os.path.exists(os.path.join(stats_folder, year)):
        os.makedirs(os.path.join(stats_folder, year))

    # Create a new CSV file
    with open(output_file, mode='w') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow([f"Summary of income for year {year}."])

        fieldnames = [col.value for col in Columns]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        months = list(year_stats.keys())
        months.sort(key=lambda x: datetime.datetime.strptime(x,'%B'))
        for month in months:
            year_stats[month].update({Columns.MONTH : month})
            writer.writerow(year_stats[month])


def load_from_csv(csv_file, columns):
    """Loads the desired columns from the csv file."""

    res_struct = defaultdict(list)
    with open(csv_file, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        metadata_rows = 2
        for row in plots:
            if metadata_rows == 2:
                res_struct['title'].append(row[0])
                metadata_rows -= 1
            elif metadata_rows:
                metadata_rows -= 1
                continue
            else:
                for column in columns:
                    col_idx = get_column_index(column)
                    # column at 0 index is string, other must be converted to float
                    if col_idx != 0:
                        res_struct[column].append(round(float(row[col_idx])) if row[col_idx] else 0)
                    else:
                        res_struct[column].append(row[col_idx])
    return res_struct


def export_two_bar_comparison(title, data, columns, labels, output_file):
    """Export the two bar comparison for given columns into .png file.
       columns is a list of column names where indivudual index represent:
       [0] = x_axix data,
       [1] = y1_data bar1,
       [2] = y2_data bar2,
       [3] = y3_data scatters.
       Respect the order of the columns.
    """

    x = np.arange(len(data[columns[0]]))
    width = 0.3
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.grid(color='0.95')
    bar1 = ax.bar(x - width / 2, data[columns[1]], color = 'salmon', width = width, label = "Wage")
    bar2 = ax.bar(x + width / 2, data[columns[2]], color = 'mediumaquamarine', width = width, label = "Haircut")
    bar3 = ax.bar(x, data[columns[3]], width = 0, color = 'cornflowerblue', label = "Sum")
    ax.scatter(x, data[columns[3]], color = 'cornflowerblue')

    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xticks(x, data[columns[0]])
    ax.legend(loc='upper left')
    ax.bar_label(bar1, padding = 3)
    ax.bar_label(bar2, padding = 3)
    ax.bar_label(bar3, padding = 3, color = 'cornflowerblue')
    fig.tight_layout()
    plt.savefig(output_file, dpi=200)


def export_plot(title, x_data, y_data, labels, output_file):
    """Export the plot for given x, y data into .png file."""

    x = np.arange(len(x_data))
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.grid(color='0.95')
    ax.plot(x, y_data, color = 'cornflowerblue', label = "Sum")
    bar = ax.bar(x, y_data, width = 0)
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xticks(x, x_data)
    ax.legend(loc='upper left')
    ax.bar_label(bar, padding = 3)
    fig.tight_layout()
    plt.savefig(output_file, dpi=200)


def export_two_bar_comparison_and_sum_plot(title, data, columns, labels, output_file):
    """Export the two bar comparison for given columns into .png file.
       columns is a list of column names where indivudual index represent:
       [0] = x_axix data,
       [1] = y1_data bar1,
       [2] = y2_data bar2,
       [3] = y3_data plot.
       Respect the order of the columns.
    """

    x = np.arange(len(data[columns[0]]))
    width = 0.3
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.grid(color='0.95')
    bar1 = ax.bar(x - width / 2, data[columns[1]], color = 'salmon', width = width, label = "Wage")
    bar2 = ax.bar(x + width / 2, data[columns[2]], color = 'mediumaquamarine', width = width, label = "Haircut")
    bar3 = ax.bar(x, data[columns[3]], width = 0)
    ax.plot(x, data[columns[3]], color = 'cornflowerblue', label = "Sum")

    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xticks(x, data[columns[0]])
    ax.legend(loc='upper left')
    ax.bar_label(bar1, padding = 3)
    ax.bar_label(bar2, padding = 3)
    ax.bar_label(bar3, padding = 3, color = 'cornflowerblue')
    fig.tight_layout()
    plt.savefig(output_file, dpi=200)


def export_year_png(year, input_file, stats_folder):
    """Creates a year statistics summary .png file from a given csv file."""

    columns = [Columns.MONTH, Columns.WAGE_EUR, Columns.HAIRCUT_EUR, Columns.INCOME_SUM]
    data = load_from_csv(input_file, columns)

    # Comparison of the wage and haircut income
    export_two_bar_comparison(
        data["title"][0],
        data,
        columns,
        ["Month", "EUR"],
        os.path.join(stats_folder, year, "Income_summary.png")
    )

    # Overall Sum plot for the current year income summary
    export_plot(
        data['title'][0],
        data[Columns.MONTH],
        data[Columns.INCOME_SUM], 
        ["Month", "EUR"],
        os.path.join(stats_folder, year, "Income_sum.png")
    )

@timing
def parse_income_data(content):
    """Parsing of the income data into desired year-based structure of dictionaries."""

    curr_month = None
    curr_year = None
    year_stats = defaultdict(dict)
    overall_stats = defaultdict(dict)
    INT_FLOAT_REGEX_GROUP = '(\d+(\.|,)?\d+)'

    # Main loop for content processing and storing in the year-based structure
    for line in content.splitlines():
        # Match a 4-digit number that represents the year
        if match := re.match(r'\b(\d{4})\b', line):
            if curr_year:
                overall_stats[curr_year] = year_stats.copy()
            curr_year = int(match.group(1))
            year_stats.clear()

        # Match newline
        elif not len(line):
            pass

        # Match czk month income value
        elif match := re.match(rf'{INT_FLOAT_REGEX_GROUP}\s*czk', line):
            czk_income = float(re.sub(",", ".", match.group(1)))
            year_stats[curr_month][Columns.WAGE_CZK.value] = czk_income

            czk_eur_rate = get_convert_rate('CZK', 'EUR', curr_month, curr_year)            
            wage_eur = round(czk_income * czk_eur_rate, 2)
            year_stats[curr_month][Columns.WAGE_EUR.value] = round(wage_eur)
            year_stats[curr_month][Columns.INCOME_SUM.value] = round(wage_eur)

        # Match eur month income value
        elif match := re.match(rf'{INT_FLOAT_REGEX_GROUP}\s*eur', line):
            wage_eur = float(re.sub(",", ".", match.group(1)))
            # Add to the EUR wage if both the czk and eur wage have been obtained.
            if Columns.INCOME_SUM.value in year_stats[curr_month]:
                year_stats[curr_month][Columns.WAGE_EUR.value] += wage_eur
            else:
                year_stats[curr_month][Columns.WAGE_EUR.value] = wage_eur
            if Columns.INCOME_SUM.value in year_stats[curr_month]:
                year_stats[curr_month][Columns.INCOME_SUM.value] += wage_eur
            else:
                year_stats[curr_month][Columns.INCOME_SUM.value] = wage_eur

        # Match eur month income value
        elif match := re.match(rf'haircuts\s+sum\s+{INT_FLOAT_REGEX_GROUP}', line):
            year_stats[curr_month][Columns.HAIRCUT_EUR.value] = float(re.sub(",", ".", match.group(1)))
            # Add the eur month income value into the income_sum place
            if Columns.INCOME_SUM.value in year_stats[curr_month]:
                year_stats[curr_month][Columns.INCOME_SUM.value] += round(year_stats[curr_month][Columns.HAIRCUT_EUR.value])
            else:
                year_stats[curr_month][Columns.INCOME_SUM.value] = round(year_stats[curr_month][Columns.HAIRCUT_EUR.value])

        # Match number of customers
        elif match := re.match(r'customers\s+count\s*(\d+)', line):
            year_stats[curr_month][Columns.NUMBER_OF_CUSTOMERS.value] = int(match.group(1))

        # Match indivudual day income in eur
        elif re.match(rf'({INT_FLOAT_REGEX_GROUP}\s*\+\s*)+{INT_FLOAT_REGEX_GROUP}', line):
            individual_days = [float(re.sub(",", ".", i)) for i in line.split() if i != '+']
            year_stats[curr_month][Columns.HAIRCUT_DAYS.value] = len(individual_days)
            year_stats[curr_month][Columns.MAX_HAIRCUT_DAY_INCOME.value] = max(individual_days)

        # Match a month name
        elif match := re.match(r'(January|February|March|April|May|June|July|August|September|October|November|December)', line):
            curr_month = match.group(1)

        else:
            logging.error(f"Skipping invalid line format: {line}")

    # Add the last processed year into the overall stats
    if year_stats:
        overall_stats[curr_year] = year_stats

    return overall_stats

def export_all_stats(overall_stats, stats_folder, output_file):
    """Export both the individual year and the overall stats into .csv-s and .png-s."""

    overall_sum = []
    months_years = []
    haircuts = []
    wages = []
    stats_overall_folder = os.path.join(stats_folder, "overall")
    # For individual year export the stats into the csv file,
    # income summary comparison and income sum into .png file.
    # And export the overall stats as well.
    for year, content in overall_stats.items():
        csv_file_name = os.path.join(stats_folder, str(year), output_file)
        export_year_csv(str(year), content, stats_folder, csv_file_name)
        export_year_png(str(year), csv_file_name, stats_folder)
        months = list(content.keys())
        months.sort(key=lambda x: datetime.datetime.strptime(x,'%B'))
        for month in months:
            months_years.append(f'{get_month_index(month, True)}/{str(year)[-2:]}')
            overall_sum.append(content[month].get(Columns.INCOME_SUM, 0))
            haircuts.append(content[month].get(Columns.HAIRCUT_EUR, 0))
            wages.append(content[month].get(Columns.WAGE_EUR, 0))

    if not os.path.exists(stats_overall_folder):
        os.makedirs(stats_overall_folder)

    # Overall Sum plot for the income summary
    export_plot(
        "Overall Sum Plot",
        months_years,
        overall_sum, 
        ["Date", "EUR"],
        os.path.join(stats_overall_folder, "Sum.png")
    )
    # Overall two bar comparison for the income summary
    export_two_bar_comparison(
        "Overall Income comparison",
        {
            "m/y": months_years,
            "wage": wages,
            "haircut": haircuts,
            "sum": overall_sum
        },
        ["m/y","wage","haircut","sum"],
        ["Date[M/Y]", "EUR"],
        os.path.join(stats_overall_folder, "Comparison.png")
    )
    # Overall two bar comparison with sum plot for the income summary
    export_two_bar_comparison_and_sum_plot(
        "Overall Sum Plot income comparison",
        {
            "m/y": months_years,
            "wage": wages,
            "haircut": haircuts,
            "sum": overall_sum
        },
        ["m/y","wage","haircut","sum"],
        ["Date[M/Y]", "EUR"],
        os.path.join(stats_overall_folder, "Sum_comparison.png")
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to the input file to be processed")
    parser.add_argument("-s", "--stats", default="stats", help="path to the overall output folder")
    parser.add_argument("output", help="path to the .csv output file for individual year statistics")
    args = parser.parse_args()

    try:
        logging.basicConfig(
            level = logging.INFO,
            format='[ %(asctime)s ][ %(levelname)s ] - %(message)s',
            datefmt='%m.%d.%Y %I:%M:%S'
        )

        logging.info(f"Income processing start!")

        logging.info(f"Loading the input file.")
        content = read_file(args.input_file)

        logging.info(f"Parsing the input data.")
        overall_stats = parse_income_data(content)

        logging.info(f"Exporting all stats.")
        export_all_stats(overall_stats, args.stats, args.output if args.output.endswith('.csv') else args.output + '.csv')

    except Exception as e:
        print(f'Following error occured: {e}')
