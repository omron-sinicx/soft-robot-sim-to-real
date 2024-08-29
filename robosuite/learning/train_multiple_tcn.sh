for seed in 0 1 2 3 4 5 6 7 8 9
do
    param1=10
    param2=5
    nohup python3 -u train_policy_tcn.py -n 02-07-tcn-circle-peg-hole-pegangle-$param1-$param2-$seed --param1 $param1 --param2 $param2 --seed $seed > nohup-$param1-$param2-$seed.out &
    # nohup python3 -u train_policy_tcn.py -n 02-06-tcn-square-peg-hole-pegangle-$param1-$param2-$seed --param1 $param1 --param2 $param2 --seed $seed > nohup-$param1-$param2-$seed.out &
    # nohup python3 -u train_policy_tcn.py -n 02-06-tcn-triangle-peg-hole-pegangle-$param1-$param2-$seed --param1 $param1 --param2 $param2 --seed $seed > nohup-$param1-$param2-$seed.out &
    # nohup python3 -u train_policy_tcn.py -n 02-06-tcn-rectangle-peg-hole-pegangle-$param1-$param2-$seed --param1 $param1 --param2 $param2 --seed $seed > nohup-$param1-$param2-$seed.out &
done