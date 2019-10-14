
for f in patrologia/output/refs/*/*; do
    sed -i.drop -E "s/\([^)(]{,50}\)/ /g" $f;
done
