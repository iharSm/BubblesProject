# BubblesProject

A methodology for testing asset price bubbles based on the theory was developed by (Robert Jarrow 2011).

#References:
1. Robert Jarrow, Younes Kchia, and Philip Protter. 2011. "How to Detect an Asset Bubble." SIAM Journal on Financial Mathematics.

#Code
Here is how to use this code

Simply call the function run_all with the following parameters

1. this is a path to a csv data file. It has to be either relative to the folder that you place
this SlidingBubbles.py file (expample below: "data/new_data/BoAFixed_FloatABSIndex.csv) or the full path (C:/blabla/blabla.csv)

2. window. this is number of data points that will be used to fit the sigma function.
3. step. this is by how much window will be shifted on each iteration
4. min_date and max_date. these are the dates that you want to limit your analysis to.
5. plot_stock. This is just a parameter that tells wether you want to plot the stock chart or not. By default it is set to no.

to recap

first iteration:
min_date                                     max_date

    |-------------------------------------------|

    |-window=256--|

second iteration:
min_date                                     max_date
    
    |-------------------------------------------|
    
    |-step=30-||-window=256--|

third iteration:
min_date                                     max_date
    
    |-------------------------------------------|
    
    |-step=30-||-step=30-||-window=256--|
