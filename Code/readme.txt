The size of a cell is DELTA x DELTA. The DELTA is the degree angles of latitude
and longitude.
  
For example, 0.001 degree or 1 mili-degree(md) of longitude is equavalent to 
82.634m at latitude 42, it is equavalent to 87.623m at latitude 38

The DELTA is an analysis parameter. 

Each cell is aligned at the DELTA.  

It is recommended to serialize the cell indexing.  

Cells without any origin and destin can be igonored.  

For each PERIOD, collect the origin-destin matrix, where origin and destin are 
cells. 

The PERIOD is an analysis parameter such as 5 min, 15 min, 20 min, 30 min, or 
1 hour. It is depending on the density of training data. 

