for dir in `ssh hasta find tmp/variant-rank-score/test_cases_explanations -type d`; do mkdir -p $dir &&  scp -r hasta:/home/tor.bjorgen/repos/cg/raredisease-ml/$dir/*.png $dir; done
