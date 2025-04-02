set terminal png size 1000,500
set datafile separator ","

set ytics
set y2tics
set xlabel "h"
set ylabel "Energy"
set y2label "Magnetization"
set grid
set key left bottom

plot \
     filename every ::1 using 2:4 with points pt 4 ps 0.85 title "Analytical E" axes x1y1,\
     filename every ::1 using 2:11 with points pt 11 ps 0.85 title "NQS E approximation" axes x1y1,\
     filename every ::1 using 2:10 with points pt 7 ps 0.85 title "Analytical Magnetization" axes x1y2,\
     filename every ::1 using 2:17 with points pt 14 ps 0.85 title "NQS Magnetization approximation" axes x1y2
set out
