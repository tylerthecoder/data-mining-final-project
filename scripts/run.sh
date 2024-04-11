outfile=results/$(date +%Y%m%d-%H%M%S).txt
python src/load-data.py | tee $outfile
