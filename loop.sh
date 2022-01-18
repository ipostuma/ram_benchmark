rm out
touch out

for i in 3 4 5 6 7
do
    for j in 64 128 256 512
    do
        SECONDS=0
        r=`python memorybenchmark.py $j $i --epochs 1`
        duration=$SECONDS
        echo $i" "$j" 10 "$duration" "$r >> out
    done
done
