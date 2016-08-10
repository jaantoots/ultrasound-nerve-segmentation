set terminal postscript eps enhanced color background "white"

inputf=sprintf("%s/accuracylog.txt", ARG1)
set output sprintf("%s/accuracy.eps", ARG1)

set multiplot layout 2,1

set ylabel "Loss"
set logscale y
plot inputf u 1:2 every ::1 w l notitle

set ylabel "Dice score"
unset logscale y
plot inputf u 1:3 every ::1 w l notitle

unset multiplot
set output
