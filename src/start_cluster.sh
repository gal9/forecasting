#!/bin/bash
clusters_file="cluster.json"
pane_name=$(tmux list-panes | grep active | cut -d':' -f 1)
echo $PANE
cp ./config.json ./config_backup.json

#tmux new-session -d -s $session_name

while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "Starting instance: $line"

    sed -i -e "s/\"sensors\":.*/\"sensors\": ${line}/" ./config_backup.json

    tmux split-pane -t $pane_name
    tmux send-keys "python main.py -c config_backup.json " $1 " " $2 " " $3 " " $4 C-m

    sleep 3

done < "$clusters_file"

rm ./config_backup.json
echo "Started cluster in tmux, session name: $session_name"