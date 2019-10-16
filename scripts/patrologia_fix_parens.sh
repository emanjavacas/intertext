
for f in output/patrologia/refs/*/*; do
    sed -i -E "s/\([^)(]{,50}\)/ /g" $f;
done
