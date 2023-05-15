The size of a cell is DELTA x DELTA. The DELTA is an analysis parameter. For example, 0.0005.
Each cell is aligned at the DELTA.  
It is recommended to serialize the cell indexing.  
Cells without any origin and destin can be igonored.  

For each PERIOD, 
	collect the origin-destin matrix, where origin and destin are cells. 

The PERIOD is an analysis parameter such as 5 min, 15 min, 20 min, 30 min, 1 hour. 	
It is depending on the density of training data. 

