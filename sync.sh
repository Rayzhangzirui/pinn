#!/bin/bash
# set -x #debug
set -e #exit on error

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
	server=ziruz16@lucy.ics.uci.edu
	remotedir=/home/ziruz16/pinn/
	;;

	*)
	echo "unknown server"
	exit 1
	;;
esac

# initialize variables for arguments
OPTIONS=""
FILE_PATTERN=""

# parse arguments
while getopts "o:f:" OPTION; do
    case $OPTION in
    o)
        OPTIONS="$OPTARG"
        ;;
    f)
        FILE_PATTERN="$OPTARG"
        ;;
    esac
done
shift $((OPTIND -1))

# current directory
cdir=${PWD%/} #need trailing / for rsync
remotedir=${remotedir%/}  # Remove trailing slash if exists


# Determine relative path
basedir=/Users/Ray/project/glioma/pinn/ # Update this to your base directory path
relpath=$(realpath --relative-to="$basedir" "$cdir")

# Append relative path to the remote directory
remotedir="$remotedir/$relpath/"

remote=$server:$remotedir #create remote dir

filelist=(--exclude={'.*','logs','__pycache__','tmp','matlabscript','sol*mat','*jpg','*png'})

# Check for optional file/pattern argument
if [ -n "$FILE_PATTERN" ]; then
    source="$cdir/$FILE_PATTERN"
else
    source=$cdir/
fi

if [ "$1" = "up" ]
then
	dest=$remote
else
	source=$remote
	dest=$cdir
fi

# https://superuser.com/questions/360966/how-do-i-use-a-bash-variable-string-containing-quotes-in-a-command
extraopt=(${@:4})

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
