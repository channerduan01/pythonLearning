# duan
sed -n "2,$1p" train.csv | cut -c 3- >data.csv
sed -n "2,$1p" train.csv | cut -c 1 >data_label.csv
sed -n "$[$1+1],\$p" train.csv | cut -c 3- >valid.csv
sed -n "$[$1+1],\$p" train.csv | cut -c 1 >valid_label.csv
sed -n '2,$p' test.csv > test_ready.csv

