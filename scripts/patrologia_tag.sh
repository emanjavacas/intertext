
ROOT=output/patrologia/refs
TARGET=output/patrologia/tagged

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
    cat $f | sed -e 's/>//g' -e 's/<//g' | bash scripts/treetagger.sh > $TARGET/$DIR/$BASE
done
