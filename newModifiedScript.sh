#!bin/bash

echo "Second one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5  --lambda_u 15 --T1 25 --T2 5 --num_epochs 300 --modified True | tee final_kmeans_overnight_50_noise

echo "Third one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5  --lambda_u 15 --T1 25 --T2 5 --num_epochs 300 --modified True --classifier gmm --hyper_parameter_type 4 | tee final_GMM_overnight_50_noise_type_4

echo "Fifth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.8  --lambda_u 25 --T1 25 --T2 5 --num_epochs 300 --modified True | tee final_kmeans_overnight_80_noise

echo "Sixth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.8  --lambda_u 25 --T1 25 --T2 5 --num_epochs 300 --modified True --classifier gmm --hyper_parameter_type 4 | tee final_GMM_overnight_80_noise_type_6