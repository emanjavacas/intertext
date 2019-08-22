
ROOT=patrologia/processed-hyphens
TARGET=patrologia/processed-tokens

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

    ucto -n $f > $TARGET/$DIR/$BASE
done
