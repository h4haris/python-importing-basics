# Python - Introduction to Importing Data in Python
Python importing data experimentation for Data Science.

Many ways to import data into Python: 
- from flat files such as .txt and .csv; 
- from files native to other software such as Excel spreadsheets, Stata, SAS, and MATLAB files; 
- and from relational databases such as SQLite and PostgreSQL.

## Introduction and flat files

### Text Files

```bash
1. Reading a text file
======================
filename = 'huck_finn.txt'
file = open(filename, mode='r') # 'r' is to read
text = file.read() 
file.close() # best practice to always close file connection
print(text)

2. Skip closing file by using Context Manager
=============================================
# using 'with' keyword
with open('huck_finn.txt', 'r') as file:
    print(file.read())

3. Use file text line by line
=============================
# using 'readline' method
with open('huck_finn.txt') as file:
    print(file.readline())
```

### Flat Files

- Text files containing records, e.g. table data
- Record: row
- Column: feature
- May contain header - check first before import/usage of data

#### Importing using NumPy

- If all the data are numerical, we can use the package numpy to import the data as a numpy array.
- Why NumPy?
    - Python standard for storing numerical data. They are efficient, fast and clean.
    - numpy arrays are often essential for other packages, such as scikit-learn
    - number of built-in functions that make it easier to import data as arrays e.g. **loadtxt()** and **genfromtxt()**

1. Using loadtxt()

```bash
1. Importing flat files using NumPy
===================================
import numpy as np
filename = 'MNIST.txt'
data = np.loadtxt(filename, delimiter=',') # default delimiter is white space
data

2. Customizing import
=====================

a. skip rows
import numpy as np
filename = 'MNIST.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1) # skip first row such as header to only load numeric data
data

b. load only specific columns
data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=[0,2]) # load only 1st & 3rd column

c. import different data types
data = np.loadtxt(filename, delimiter=',', dtype=str) # all entries will be imported as strings

d. import tab-delimited file
data = np.loadtxt(filename, delimiter='\t') # for tab-delimited
```

2. Using genfromtxt()
- In np.genfromtxt(), If we pass **dtype=None** to it, it will figure out what types each column should be.
- Because numpy arrays have to contain elements that are all the same type, the structured array solves this by being a 1D array, where each element of the array is a row of the flat file imported e.g. (100,)

```bash
1. Importing flat files 
=======================
import numpy as np
data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None) # names tells us there is a header

2. Accessing rows & columns
===========================
data[1] # to access 2nd row
data['Fare'] # to access Fare column data from all row
```

3. Using genfromtxt()
- behaves similarly to np.genfromtxt(), except that its default dtype is None, in addition to defaults delimiter=',' and names=True. 

```bash
d = np.recfromcsv('titanic.csv')
print(d[:2]) # print first 2 entries
```

#### Importing using Pandas

- Why Pandas?
    - provide 2D labeled data structures.
    - columns of different types
    - manipulate, slice, reshape, groupby, join, merge
    - perform statistics
    - work with time series data
- What problems does Pandas solve?
    - Python has long been great for data munging and preparation, but less so for data analysis and modeling. 
    - pandas helps fill this gap, enabling us to carry out our entire data analysis workflow in Python without having to switch to a more domain specific language like R.
- Manipulating pandas DataFrames - useful in all steps of the data scientific method
    - exploratory data analysis 
    - data wrangling, 
    - data preprocessing, 
    - building models and 
    - visualization. 
- standard and best practice in Data Science to use pandas to import flat files as dataframes.

```bash
1. Importing using Pandas
=========================
import pandas as pd
filename = 'winequality-red.csv'
data = pd.read_csv(filename)
data. head()

2. Convert dataframe to numpy arrays
====================================
data_array = data.values

3. Customizing import
=====================

a. import the first n rows
data = pd.read_csv(file, nrows=5) # use nrows to import the first n rows

b. skip header
data = pd.read_csv(file, nrows=5, header=None) # use header=None if there is no header in file

c. change delimiter
data = pd.read_csv(file, sep='\t') # use sep argument for defining delimiter

d. remove comments from columns
# comment takes characters that comments occur after in the file, which in here is '#'
data = pd.read_csv(file, sep='\t', comment='#') 

e. handle missing values
# na_values takes a list of strings to recognize as NA/NaN, here the string 'Nothing'
data = pd.read_csv(file, sep='\t', comment='#', na_values='Nothing') 
```

**Feather - a new and fast on-disk format for dataframes for R and Python**

## Importing data from other file types

### Files Types include:
- pickled files
- Excel spreadsheets, 
- SAS and Stata files, 
- HDF5 files - a file type for storing large quantities of numerical data, and 
- MATLAB files

#### Pickled files
- file type native to Python
- used for many data types for which it is not obvious how to store them
- pickled files are serialized
- Serialize = converting the object into a sequence of bytes, or bytestream


```bash
Importing pickle file
=====================
import pickle
with open('pickled_fruit.pkl', 'rb') as file: # 'rb' for read-only binary file
    data = pickle.load(file)
print(data)

Output: {'peaches': 13, 'apples':4}
```

#### Excel files

```bash
1. Importing Excel spreadsheets
===============================
import pandas as pd
file = 'urbanpop.xlsx'
data = pd.ExcelFile(file)
print (data.sheet_names)

Output: ['1960-1966', '1967-1974']

2. Read sheet data
==================
df1 = data.parse('1960-1966') # sheet name, as a string
df2 = data.parse(0) # sheet index, as a float

3. Customizing excel import
===========================
# skiprows - provide specific row numbers to skip in list
# names - provide name the columns as strings in list
# usecols - provide specific columns to parse as column numbers in list

a. skip rows
df1 = xls.parse(0, skiprows=[0]) # skip first row

b. import certain columns
df2 = xls.parse(1, usecols=[0], skiprows=[0], names=['Country']) # Parse the first column only

c. change column names
df1 = xls.parse(0, skiprows=[0], names=['Country', 'AAM due to War (2002)']) # change column names with provided names
```

#### Checking list directories
```bash
import os
wd = os.getcwd() # get name of the current directory
os.listdir(wd) # list contents of the directory
```

#### SAS/Stata files
- SAS: 
    - **Statistical Analysis System**
    - popular for business analytics and biostatistics
    - important because SAS is a software suite that performs:
        - advanced analytics, 
        - multivariate analyses, 
        - business intelligence, 
        - data management, 
        - predictive analytics 
        - standard for computational analysis
    - most common SAS file have extension 
        - **.sas7bdat** for dataset files & 
        - **.sas7bcat** for catalog files
- Stata: 
    - **"Statistics" + "data"**
    - academic social sciences research, such as economics and epidemiology
    - extension with **.dta**

```bash
1. Importing SAS files
======================
import pandas as pd
from sas7bdat import SAS7BDAT

with SAS7BDAT('urbanpop.sas7bdat') as file:
    df_sas = file.to_data_frame()

2. Importing Stata files
========================
import pandas as pd
data = pd.read_stata('urbanpop.dta')
```

#### HDF5 files
- **Hierarchical Data Format version 5**
- standard for storing large quantities of numerical data
- Datasets can be of hundreds of GBs or TBs
- HDF5 can scale to exabytes
- HDF project is actively managed by HDF Group

```bash
Importing HDF5 files
====================
import h5py
filename = 'H-H1_LOSC_4_V1-815411200-4096.hdf5'
data = h5py.File(filename, 'r') # 'r' is to read
print (type(data))
```

Structure of HDF5 files
- hierarchical nature of the file structure
- three groups as directories:
    - meta - contains meta-data for the file 
    - quality - contains information about data quality
    - strain - contains the data of interest
- Each of these is an HDF group.

```bash
1. View structure of file
=========================
for key in data.keys():
    print(key)

Output: 
meta
quality
strain

2. Accessing a Group
====================
print(type(data['meta']))

3. Exploring MetaData Keys
==========================
for key in data['meta'].keys():
    print(key)

Output Example:
Description
Duration
Type

4. Accessing MetaData Keys
==========================
# converting needed keys to numpy arrays
print(np.array(data['meta']['Description']), np.array(data[ ‘meta’ ]['Detector']))
```

#### MATLAB files
- short for **Matrix Laboratory**
- industry standard in engineering and science. 
- Data saves as **'.mat'** files
- python library scipy has functions to read and write .mat files
    - scipy.io.loadmat() - read .mat file
    - scipy.io.savemat() - write .mat file

**.mat files** - A .mat file is simply a collection of objects from MATLAB workspace where all variables are stored, such as strings, floats, vectors and arrays, among many other objects.

```bash
Importing MATLAB files
======================
import scipy.io
filename = 'workspace.mat'
mat = scipy.io.loadmat(filename)
print(type(mat))

Output: dict
```

keys = MATLAB variable names;
values = object assigned to variables

```bash
Accessing a variable
====================
print(mat['x'])
print(type(mat['x']))
```

## Working with relational databases in Python

### Relational Model
- widely adopted
- Follow 13 Codd's Rules, which defined to describe what a RDBMS as relational

### SQLite
- SQLite database - fast and simple
- packages to access SQLite DB are **sqlite3** and **SQLAlchemy**

```bash
1. Creating a database engine
=============================
from sqlalchemy import create_engine
engine = create_engine('sqlite:///Northwind.sqlite')

2. Getting all table names
==========================
table_names = engine.table_names()
print(table_names)

3. Querying a table
===================

a. without context manager
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///Northwind.sqlite')
con = engine.connect()
rs = con.execute("SELECT * FROM Orders") # rs is SQLAlchemy results object
df = pd.DataFrame(rs.fetchall()) # turn rs into dataframe
df.columns = rs.keys() # to set table headers in dataframe columns
con.close()


b. with context manager
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')

with engine.connect() as con:
    rs = con.execute("SELECT OrderID, OrderDate FROM Orders")
    df = pd.DataFrame(rs.fetchmany(size=5)) # using fetchmany importing 5 rows instead of all rows
    df.columns = rs.keys()

c. with pandas method [EFFICIENT ONE LINER]
df = pd.read_sql_query("SELECT * FROM Orders", engine)


4. Joining tables
=================
df = pd.read_sql_query("SELECT OrderID, CompanyName FROM Orders
                        INNER JOIN Customers on Orders.CustomerID = Customers.CustomerID", engine)
print(df.head())
```