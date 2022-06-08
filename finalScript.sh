#!/bin/bash

echo "First one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 25 --T2 5 --num_epochs 300 --modified True --classifier gmm | tee final_GMM_overnight_1

echo "Second one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 25 --T2 5 --num_epochs 300 --modified True | tee final_kmeans_overnight_1

echo "Third one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 25 --T2 5 --num_epochs 300 --classifier gmm --hyper_parameter_type 2 | tee final_GMM_overnight_2

echo "Fourth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 25 --T2 5 --num_epochs 300 --classifier gmm --hyper_parameter_type 3 | tee final_GMM_overnight_3

echo "Fifth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 25 --T2 5 --num_epochs 300 --classifier gmm --hyper_parameter_type 4 | tee final_GMM_overnight_4

echo "Sixth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 25 --T2 5 --num_epochs 300 --classifier gmm --hyper_parameter_type 5 | tee final_GMM_overnight_5

echo "Seventh one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 25 --T2 5 --num_epochs 300 --classifier gmm --hyper_parameter_type 6 | tee final_GMM_overnight_6

echo "Eighth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 25 --T2 5 --num_epochs 300 --classifier gmm --hyper_parameter_type 7 | tee final_GMM_overnight_7

echo "Nineth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 25 --T2 5 --num_epochs 300 --classifier gmm --hyper_parameter_type 8 | tee final_GMM_overnight_8
