#/bin/bash
root=/public/home/yuqi/lawliet/nlp/wentian/
dir=03-23-15-03

date
echo $root
cd $root
echo ./logs/$dir/output/
python data_check.py --path $dir
cd ./logs/$dir/output/
tar zcvf $dir.tar.gz doc_embedding query_embedding

date