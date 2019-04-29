sleep 10 && echo "hello"
sleep 5 && sh copy_result.sh  # the second line must wait until the first line finished, so do this every 1 hour
sleep 1h && sh copy_result.sh
sleep 1h && sh copy_result.sh
sleep 1h && sh copy_result.sh
sleep 1h && sh copy_result.sh
