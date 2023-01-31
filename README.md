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
Note that the `output_file` will by the new name for the `.csv` file. So if you do not add the `.csv` file ending the script will handle that.
