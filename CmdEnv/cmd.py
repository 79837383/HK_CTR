mkdir -p output; python avazu_data_processer.py --data_path train.txt --output_dir output --num_lines_to_detect 1000 --test_set_size 100 --batch_size 100

python train.py --train_data_path ./output/train.txt --test_data_path ./output/test.txt --data_meta_file ./output/data.meta.txt --model_type=0  --batch_size 100



python infer.py --model_gz_path xx.tar.gz --data_path ./output/infer.txt --prediction_output_path predictions.txt --data_meta_path ./output/data.meta.txt --model_type=0
