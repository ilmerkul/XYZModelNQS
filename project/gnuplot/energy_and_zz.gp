
set terminal png size 1500,500
set datafile separator ","

set ytics
set y2tics
set xlabel "h"
set ylabel "Energy"
set y2label "ZZ correlations"
set grid
set key outside box

plot \
     filename every ::1 using 2:4 with points pt 4 ps 2 lc rgb "blue" title "Analytical E" axes x1y1,\
     filename every ::1 using 2:11 with points pt 1 ps 2 lc rgb "black" title "NQS E approximation" axes x1y1,\
     filename every ::1 using 2:7 with points pt 8 ps 2 lc rgb "blue" title "Analytical ZZ" axes x1y2,\
     filename every ::1 using 2:16 with points pt 2 ps 2 lc rgb "black" title "NQS ZZ" axes x1y2
set out
