#!/bin/bash

echo "First one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 200 --T2 5 --num_epochs 300 --modified True --classifier gmm | tee overnight_gmm_20

echo "Second one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 200 --T2 5 --num_epochs 300 --modified True | tee overnight_kmeans_20

echo "Third one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 200 --T2 5 --num_epochs 300 | tee overnight_normal_20

echo "Fourth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5  --lambda_u 15 --T1 200 --T2 5 --num_epochs 300 --modified True --classifier gmm | tee overnight_gmm_50

echo "Fifth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5  --lambda_u 15 --T1 200 --T2 5 --num_epochs 300 --modified True | tee overnight_kmeans_50

echo "Sixth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5  --lambda_u 15 --T1 200 --T2 5 --num_epochs 300 | tee overnight_normal_50

echo "Seventh one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.8  --lambda_u 25 --T1 200 --T2 5 --num_epochs 300 --modified True --classifier gmm | tee overnight_kmeans_80

echo "Eighth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.8  --lambda_u 25 --T1 200 --T2 5 --num_epochs 300 --modified True | tee overnight_kmeans_80

echo "Nineth one"
python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.8  --lambda_u 25 --T1 200 --T2 5 --num_epochs 300 | tee overnight_normal_80

echo "---------------------------------"
echo "---------------------------------"
echo "---------------------------------"

echo "First one"
python PES_semi.py --dataset cifar100 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 200 --T2 5 --num_epochs 300 --modified True --classifier gmm | tee overnight_gmm_20_cifar100

echo "Second one"
python PES_semi.py --dataset cifar100 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 200 --T2 5 --num_epochs 300 --modified True | tee overnight_kmeans_20_cifar100

echo "Third one"
python PES_semi.py --dataset cifar100 --noise_type symmetric --noise_rate 0.2  --lambda_u 5 --T1 200 --T2 5 --num_epochs 300 | tee overnight_normal_20_cifar100

echo "Fourth one"
python PES_semi.py --dataset cifar100 --noise_type symmetric --noise_rate 0.5  --lambda_u 15 --T1 200 --T2 5 --num_epochs 300 --modified True --classifier gmm | tee overnight_gmm_50_cifar100

echo "Fifth one"
python PES_semi.py --dataset cifar100 --noise_type symmetric --noise_rate 0.5  --lambda_u 15 --T1 200 --T2 5 --num_epochs 300 --modified True | tee overnight_kmeans_50_cifar100

echo "Sixth one"
python PES_semi.py --dataset cifar100 --noise_type symmetric --noise_rate 0.5  --lambda_u 15 --T1 200 --T2 5 --num_epochs 300 | tee overnight_normal_50_cifar100

echo "Seventh one"
python PES_semi.py --dataset cifar100 --noise_type symmetric --noise_rate 0.8  --lambda_u 25 --T1 200 --T2 5 --num_epochs 300 --modified True --classifier gmm | tee overnight_kmeans_80_cifar100

echo "Eighth one"
python PES_semi.py --dataset cifar100 --noise_type symmetric --noise_rate 0.8  --lambda_u 25 --T1 200 --T2 5 --num_epochs 300 --modified True | tee overnight_kmeans_80_cifar100

echo "Nineth one"
python PES_semi.py --dataset cifar100 --noise_type symmetric --noise_rate 0.8  --lambda_u 25 --T1 200 --T2 5 --num_epochs 300 | tee overnight_normal_80_cifar100

