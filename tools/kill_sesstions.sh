#!/bin/bash

# リストされている全ての screen セッションを取得して終了させる
for session in $(screen -ls | grep -oP '\d+\.\w+' | cut -d. -f1); do
    screen -S "${session}" -X quit
done
