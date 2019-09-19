
ROOT=patrologia/output/refs
TARGET=patrologia/output/tokenized

if [ ! -e $TARGET ]; then
    mkdir $TARGET;
fi

for f in $ROOT/*/*; do
    DIR=`dirname $f`
    DIR=`basename $DIR`
    BASE=`basename $f`
    if [ ! -e $TARGET/$DIR ]; then
	mkdir $TARGET/$DIR;
    fi

    ucto -S -n $f > $TARGET/$DIR/$BASE
done
