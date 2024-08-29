param1=10
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for param2 in 5
    do
        nohup python3 -u train_student.py learning_progress/01-19-circle-noalignment-peg-hole-pegangle-seed-10-$param2-$seed/latest.pt -n 01-28-circle-noalignment-peg-hole-pegangle-seed-10-$param2-$seed-student --param1 $param1 --param2 $param2 > nohup-circle-$seed-$param2-1.out &
        # nohup python3 -u train_student.py learning_progress/01-19-square-peg-hole-pegangle-seed-10-$param2-$seed/latest.pt -n 02-05-square-peg-hole-pegangle-seed-10-$param2-$seed-student --param1 $param1 --param2 $param2 > nohup-square-$seed-$param2-1.out &
        # nohup python3 -u train_student.py learning_progress/01-19-triangle-peg-hole-pegangle-seed-10-$param2-$seed/latest.pt -n 02-05-triangle-peg-hole-pegangle-seed-10-$param2-$seed-student --param1 $param1 --param2 $param2 > nohup-triangle-$seed-$param2-1.out &
        # nohup python3 -u train_student.py learning_progress/01-19-rectangle-peg-hole-pegangle-seed-10-$param2-$seed/latest.pt -n 02-05-rectangle-peg-hole-pegangle-seed-10-$param2-$seed-student --param1 $param1 --param2 $param2 > nohup-rectangle-$seed-$param2-1.out &
    done
done