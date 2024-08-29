for param1 in 10
do
    for param2 in 5
    do
        for seed in 0 1 2 3 4 5 6 7 8 9
        do
            nohup python3 -u train_policy.py -n 01-19-circle-noalignment-peg-hole-pegangle-seed-$param1-$param2-$seed --param1 $param1 --param2 $param2 --seed $seed > nohup-circle-$param1-$param2-$seed.out &
            # nohup python3 -u train_policy.py -n 01-31-square-torque-peg-hole-pegangle-seed-$param1-$param2-$seed --param1 $param1 --param2 $param2 --seed $seed > nohup-square-$param1-$param2-$seed.out &
            # nohup python3 -u train_policy.py -n 01-31-triangle-torque-peg-hole-pegangle-seed-$param1-$param2-$seed --param1 $param1 --param2 $param2 --seed $seed > nohup-triangle-$param1-$param2-$seed.out &
            # nohup python3 -u train_policy.py -n 01-31-rectangle-torque-peg-hole-pegangle-seed-$param1-$param2-$seed --param1 $param1 --param2 $param2 --seed $seed > nohup-rectangle-$param1-$param2-$seed.out &
        done
    done
done