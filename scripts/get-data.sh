mkdir -p data/raw
mkdir -p data/zips 
mkdir -p data/csv

curl "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/4651/35131/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1713024836&Signature=dUD2WQbc0FUPcZfhCavaRDfnafTb0koYYDcKMXPufFfxxuSjqL3mndP0A5it2mIJq0qfGX0OawKrqyNjL%2FVeAPqkcKgj6UASdsQBp5NbSD3Cj9A%2FLLjHYEKB%2FdaNWeCQna1SXlyFZ4h6WdVu5x%2Ba4LyovZPwHI%2FpUVg55L%2B8L91CcGHaf2bhGpxp5He77OhAnjqR2ZTPKBC9DOKeOVJhhvCdxSyjfG5mLRJn7lYvd619ZZpsc3EiUwdQwDhY9pAzIChChRSpXwxmHlgHyiuIAoFmKaFAR4H3XPRkPt%2BVq8%2FweJgF97L06HuwQkGS1M55Yj%2B7zV1WirikFtbd%2FyfERg%3D%3D&response-content-disposition=attachment%3B+filename%3Dairbnb-recruiting-new-user-bookings.zip" -o data/raw/data.zip

unzip data/raw/data.zip -d data/zips

# unzip every file in `data/zips` into `data/csv`
for file in data/zips/*.zip
    do
        unzip $file -d data/csv
    done

rm -rf data/raw
rm -rf data/zips

