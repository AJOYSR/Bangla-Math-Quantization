#!/usr/bin/env bash
# Build both versions of the paper from main.tex.
#   paper_Sarker_Riya.pdf            with author names (preprint, CV, GitHub)
#   paper_anonymous_submission.pdf   anonymous, with line numbers (for review)
set -euo pipefail
cd "$(dirname "$0")"

tectonic main.tex
mv main.pdf paper_anonymous_submission.pdf

sed 's/\\usepackage\[review\]{acl}/\\usepackage[preprint]{acl}/' main.tex > main_preprint.tex
tectonic main_preprint.tex
mv main_preprint.pdf paper_Sarker_Riya.pdf
rm main_preprint.tex

echo "Built paper_Sarker_Riya.pdf and paper_anonymous_submission.pdf"
