# Optimized-EWMA

This project aims to convert statistical method, i.e, Exponential Weighted Moving Average(EWMA) of time series forecasting into a swarm intelligence method by using metaheuristic techniques such as Particle Swarm Optimization(PSO), Gravitational Search Algorithm(GSA) and Genetic Algorithm(GA). 

**Data**  
There are several time series datasets available in this folder on which results of experiment are tested.

**Code**  
In the ewma.py file, PSO, GSA & GA are used to find value of parameters of EWMA such that error will be minimum. Moreover, EWMA is implemented by considering these values of parameters and finally metaheuristic techniques are compared.

**Other Existing Methods**  
In this folder, existing methods of time series forecasting such as Double Smooting, Holt-Winter Additive & Holt-Winter Multiplicative are compared with EWMA best metaheuristic technique.

Efficient Algorithm to Find Optimum Window Size for Time Series Forecasting

 Using Particle Swarm Optimization, an algorithm was found to efficiently pre-process a time series and accurately predict both the window size and weights for moving average techniques such as SMA, WMA and EWMA. Using these parameters, the future values of the time series were predicted.
Through experimental results, this algorithm was proved to be much more efficient than the ones currently in use, such as the Straightforward Method, Gravitational Search Algorithm and Genetic Algorithm.
