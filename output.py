# checking the output file from cprofile.

#tottime seems to be the time spent inside the function and cumtime seem to be total time spent including calls to other functions
#as cumtime is always larger. So they are equal when the function does not call any other functions. 
import pstats

p = pstats.Stats('output.txt')
p.sort_stats('cumulative').print_stats(50)
