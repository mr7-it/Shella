# WordNet Bahasa r24

Source: `open-language/wordnets` commit
`fb201ec6155dc5c3a23d76673cfbb28630bc9a67`, files under
`data/msa-wordnet-r24/`.

The data license was checked against the original SourceForge project
file `https://sourceforge.net/p/wn-msa/tab/HEAD/tree/trunk/LICENSE?format=raw`.
That notice explicitly covers the Wordnet Bahasa data and identifies it
as MIT. The verbatim notice is in `LICENSE`.

Shella imports only Indonesian (`I`) entries with quality `Y` (hand
checked and good). It excludes Malay (`M`), Bahasa-wide (`B`), and all
other quality levels. Synset identifiers are the upstream PWN 3.0
offset-POS identifiers; Shella does not import or redistribute PWN data
or infer hypernym/meronym edges from those identifiers. The upstream
README estimates 5-10% errors across the release and cautions against
using low-quality entries.