# LSD
Official Release of "Learning the Stein Discrepancy for Training and Evaluating Energy-Based Models without Sampling"

## To run toy data:
python lsd_toy.py --save /tmp/test_release --data moons --base_dist

## To run ICA:
### mle
python lsd_ica.py --test_freq 1000 --dim ${DIM} --mode mle --save ${SAVE} --lr ${LR} --batch_size 1000 --test_batch_size 1000 --niters 100000 --log_freq 100 --seed 1235
### score matching
python lsd_ica.py --test_freq 1000 --dim ${DIM} --mode sm --save ${SAVE} --lr ${LR} --batch_size 1000 --test_batch_size 1000 --niters 100000 --log_freq 100 --seed ${SEED}
### nce
python lsd_ica.py --test_freq 1000 --dim ${DIM} --mode nce --save ${SAVE} --lr ${LR} --batch_size 1000 --test_batch_size 1000 --niters 100000 --log_freq 100 --seed ${SEED}
### cnce (recommended EPS in (.01 .1 1.))
python lsd_ica.py --test_freq 1000 --dim ${DIM} --mode cnce-${EPS} --save ${SAVE} --lr ${LR} --batch_size 1000 --test_batch_size 1000 --niters 100000 --log_freq 100 --seed ${SEED} 
### LSD (recommended KITER in (1, 5) L2 in (.01, .1, 1., 10.)
python lsd_ica.py --test_freq 1000 --dim ${DIM} --mode functional-${L2} --save ${SAVE} --lr ${LR} --batch_size 1000 --test_batch_size 1000 --niters 100000 --log_freq 100 --k_iters ${KITER} --seed ${SEED} &


## To run MNIST (recommended L2 (10., 1., .1) LR in (.0001, .00001, .001)
python lsd_mnist.py --lr ${LR} --batch_size 256 --l2 ${L2} --save ${SAVE} --k_iters 5 --e_iters 1 --n_steps 100 --epochs 100 --viz_freq 500 --arch mlp --logit --weight_decay .0005 --base_dist

## To run hypothesis testing
must pip install git+https://github.com/wittawatj/kernel-gof.git and pip install git+https://github.com/wittawatj/interpretable-test
### LSD
python lsd_test.py --test rbm-pert --sigma_pert ${PERT} --n_iters 1000 --l2 .5 --batch_size 800 --weight_decay .0005 --seed ${SEED} --n_train 800 --n_val 100 --n_test 100 --save ${SAVE} --dropout --maximize_power --val_power
### FSSD
python lsd_test.py --test rbm-pert --sigma_pert ${PERT} --seed ${SEED} --n_train 200 --n_val 0 --n_test 800 --save ${SAVE} --test_type fssd
### MMD
python lsd_test.py --test rbm-pert --sigma_pert ${PERT} --seed ${SEED} --n_train 200 --n_val 0 --n_test 800 --save ${SAVE} --test_type mmd
### LKSD
python lsd_test.py --test rbm-pert --sigma_pert ${PERT} --seed ${SEED} --n_train 200 --n_val 0 --n_test 800 --save ${SAVE} --test_type lksd
### KSD
python lsd_test.py --test rbm-pert --sigma_pert ${PERT} --seed ${SEED} --n_train 200 --n_val 0 --n_test 800 --save ${SAVE} --test_type ksd
