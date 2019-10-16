
# download input data
echo "Downloading patrologia data"
mkdir -p source/patrologia

wget -qO- https://github.com/OpenGreekAndLatin/patrologia_latina-dev/archive/master.tar.gz | tar -xzC source/patrologia -f - patrologia_latina-dev-master/corrected --strip-components=2

# from source/patrologia to output/patrologia/raw/
echo "Extracting raw text"
python intertext/patrologia/extract_text.py

# from output/patrologia/raw to output/patrologia/merged
echo "Fixing hyphenations"
python intertext/patrologia/merge_splits.py

# from output/patrologia/merged to output/patrologia/refs
echo "Detecting references"
python intertext/patrologia/detect_refs.py

echo "Dropping parenthesized info"
bash scripts/patrologia_fix_parens.sh

echo "Tagging the patrologia with treetagger"
bash scripts/patrologia_tag.sh
