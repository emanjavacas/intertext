#!/bin/sh

# Set these paths appropriately

BIN="/home/manjavacas/code/vendor/treetagger/bin"
CMD="/home/manjavacas/code/vendor/treetagger/cmd"
LIB="/home/manjavacas/code/vendor/treetagger/lib"

OPTIONS="-token -lemma -sgml -cap-heuristics"

TOKENIZER=${CMD}/utf8-tokenize.perl
MWL=${CMD}/mwl-lookup.perl
TAGGER=${BIN}/tree-tagger
ABBR_LIST=intertext/patrologia/latin.abbrv
PARFILE=${LIB}/latin.par
MWLFILE=${LIB}/latin-mwls

$TOKENIZER -a $ABBR_LIST $* |
# recognition of MWLs
$MWL -f $MWLFILE |
# tagging
$TAGGER $OPTIONS $PARFILE

