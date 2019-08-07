
INPUT=corrected
OUTPUT=processed

if [ ! -d $OUTPUT ]; then
    mkdir $OUTPUT
fi

XSL=passage.transform.xsl

for dir in $INPUT/*; do
    output_dir=$OUTPUT/`basename $dir`
    if [ ! -d $output_dir ]; then
	mkdir $output_dir
    fi
    for f in $dir/*; do
	output_file=$output_dir/`basename $f`
    	xsltproc $XSL $f | python fix_hyphens.py | python map_chars.py > $output_file
    done
done
