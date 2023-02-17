#!/bin/bash
# set -x #debug
if [ "$1" != "up" ]&& [ "$1" != "down" ]
then
	echo "up or down"
	exit 1
fi

case "$2" in
	"gru")
	server=rzhang@gru.ics.uci.edu
	remotedir=/home/rzhang/pinn/
	;;

	"hpc")
	server=ziruz16@hpc3.rcic.uci.edu
	remotedir=/data/homezvol0/ziruz16/pinn/
	;;

	"opengpu")
	server=ziruz16@opengpu.ics.uci.edu
	remotedir=/home/ziruz16/pinn/
	;;

	*)
	echo "unknown server"
	exit 1
	;;
esac

# current directory
cdir=$PWD/ #need trailing / for rsync

remote=$server:$remotedir #create remote dir

filelist=(--exclude={'.*','logs','__pycache__','tmp','matlabscript','sol*mat','*jpg','*png'})
if [ "$1" = "up" ]
then
	dest=$remote
	source=$cdir
else
	source=$remote
	dest=$cdir
fi

# https://superuser.com/questions/360966/how-do-i-use-a-bash-variable-string-containing-quotes-in-a-command
extraopt=(${@:3})


echo "source = $source, dest = $dest"

# echo $filelist

cmd="rsync -nrtuv ${source} ${dest} ${filelist[@]} ${extraopt[@]}"
cmd2="rsync -rtuv ${source} ${dest} ${filelist[@]} ${extraopt[@]}"
echo $cmd
eval $cmd

echo "Ready to excecute:  $cmd2"

# https://stackoverflow.com/questions/226703/how-do-i-prompt-for-yes-no-cancel-input-in-a-linux-shell-script
while true; do
    read -p "Continue? " yn
    case $yn in
        [Yy]* ) eval $cmd2; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done