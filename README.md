# Income processor

Python script for plain text income file processing and export of yearly-based and overall income data.
Exported files consist of `.csv` and `.png` files.

## Input file sturcture
Script takes as the input a plain text file which processes sequencally be lines.

## Usage
Script runs on the `Python 3.11` because of the usage of `StrEnum` feature needed for columns names representation.

Once the execution rights are granted just run the script:

```
./income_processing.py input_file output_file
```
Where the `input_file` refers to your input file name or paths to the file.

**_NOTE:_ The `output_file` will by the new name for the `.csv` file. So if you do not add the `.csv` file ending the script will handle that.**

## Example input file

Essential is the date values desciption in the file. Where the `year` and `month` are mandatory.
Into each month you can add following data:

 - `<xy> czk`
 - `<xy> eur`
 - `haircut sum <xy>`
 - `<x> + <y> + ...`
 - `customers count: <xy>`

Example input file could look like this:
```
2023
January
112 + 81 + 74
haircuts sum 267
customers count: 20

February
85000 czk
1000 eur

```
**_NOTE:_ All value fields are optional.**
