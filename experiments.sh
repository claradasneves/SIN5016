#!/bin/bash

# script para disparar todos os experimentos

## Modelos s/ HOG
# experimento 1:

# experimento 2: 
echo "running mlp, regularizer=l2, optimizer=GD"
python main_true.py \
--features-dir data/hog_npy \
--model mlp \
--regularization l2 \
--optimizer gd \
--num-classes 50 > experiments/logs/mlp+reg=l2+opt=gd-run_error.txt

# experimento 3: mlp, regularizer=None, optimizer=Newton

# experimento 4: mlp, regularizer=l2, optimizer=Newton
echo "running mlp, regularizer=l2, optimizer=Newton"
python main_true.py \
--features-dir data/hog_npy \
--model mlp \
--regularization l2 \
--optimizer newton \
--num-classes 50

# experimento 5: reglog, regularizer=None, optimizer=GD

# experimento 6: reglog, regularizer=l2, optimizer=GD
echo "running reglog, regularizer=l2, optimizer=GD"
python main_true.py \
--features-dir data/hog_npy \
--model reglog \
--regularization l2 \
--optimizer gd \
--num-classes 50

# experimento 7: reglog, regularizer=None, optimizer=Newton

# experimento 8: reglog, regularizer=l2, optimizer=Newton
echo "running reglog, regularizer=l2, optimizer=Newton"
python main_true.py \
--features-dir data/hog_npy \
--model reglog \
--regularization l2 \
--optimizer newton \
--num-classes 50

## Modelos c/ HOG
# experimento 10: mlp+hog, regularizer=None, optimizer=GD

# experimento 11: mlp+hog, regularizer=l2, optimizer=GD
echo "running mlp+hog, regularizer=l2, optimizer=GD"
python main_true.py \
--features-dir data/hog_npy \
--model mlp \
--regularization l2 \
--optimizer gd \
--num-classes 50 \
--use-attributes \
--attr-file data/atributos/list_attr_celeba.csv

# experimento 12: mlp+hog, regularizer=None, optimizer=Newton

# experimento 13: mlp+hog, regularizer=l2, optimizer=Newton
echo "running mlp+hog, regularizer=l2, optimizer=Newton"
python main_true.py \
--features-dir data/hog_npy \
--model mlp \
--regularization l2 \
--optimizer newton \
--num-classes 50 \
--use-attributes \
--attr-file data/atributos/list_attr_celeba.csv

# experimento 14: reglog+hog, regularizer=None, optimizer=GD

# experimento 15: reglog+hog, regularizer=l2, optimizer=GD
echo "running reglog+hog, regularizer=l2, optimizer=GD"
python main_true.py \
--features-dir data/hog_npy \
--model reglog \
--regularization l2 \
--optimizer gd \
--num-classes 50 \
--use-attributes \
--attr-file data/atributos/list_attr_celeba.csv

# experimento 16: reglog+hog, regularizer=None, optimizer=Newton

# experimento 17: reglog+hog, regularizer=l2, optimizer=Newton
echo "running reglog+hog, regularizer=l2, optimizer=Newton"
python main_true.py \
--features-dir data/hog_npy \
--model reglog \
--regularization l2 \
--optimizer newton \
--num-classes 50 \
--use-attributes \
--attr-file data/atributos/list_attr_celeba.csv