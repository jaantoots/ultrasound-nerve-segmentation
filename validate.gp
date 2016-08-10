set terminal postscript eps enhanced color background "white"

inputf=sprintf("%s/validate.sort.tsv", ARG1)
set output sprintf("%s/validate.eps", ARG1)

set xlabel "Iteration"
set ylabel "Dice score"
plot inputf u 1:2 every ::1 w lp title "Train", inputf u 1:3 every ::1 w lp title "Validate"

set output
