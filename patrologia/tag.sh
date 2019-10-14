
ROOT=patrologia/output/refs
TARGET=patrologia/output/tagged

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
    cat $f | sed -e 's/>//g' -e 's/<//g' | bash patrologia/treetagger.sh > $TARGET/$DIR/$BASE
done
