echo "=== Transformer on IWSLT==="

DATA_PATH=./data-bin/iwslt14.tokenized.de-en.joined
ARCH=transformer_iwslt_de_en_v2
alg=adam
lr=0.0015
eps=1e-8
seed=0
use_warmup=1

init_method=none # ['none','random','random-kaiming','grad-mean','grad-sq','grad-var','grad-mean-var', 'grad-mean-random']
init_state='mv'
init_scale=1.0
init_scale_m0=1.0


init_method='none'
init_state='v'
export CUDA_VISIBLE_DEVICES=0;
for init_scale in 1
do
python -u  main.py --data ${DATA_PATH} --seed $seed --optimizer $alg --adam-betas '(0.9, 0.98)' --adam-eps $eps \
   --lr $lr --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
  --arch ${ARCH} --share-all-embeddings  --clip-norm 0.0  --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
  --max-tokens 4096  --update-freq 1 --no-progress-bar --log-interval 50 --keep-last-epochs 5 --max-epoch 55 \
  --init_method $init_method --init_state $init_state --init_scale $init_scale --init_scale_m0 $init_scale_m0 --use_warmup $use_warmup  \
  > nohups/exp_transformer-${alg}-${lr}-${eps}-${seed}_init${init_method}-${init_state}-${init_scale}-${init_scale_m0}_${use_warmup}.log 2>&1 &
done


init_method='random'
init_state='v'
export CUDA_VISIBLE_DEVICES=1;
for init_scale in 1 100
do
python -u  main.py --data ${DATA_PATH} --seed $seed --optimizer $alg --adam-betas '(0.9, 0.98)' --adam-eps $eps \
   --lr $lr --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
  --arch ${ARCH} --share-all-embeddings  --clip-norm 0.0  --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
  --max-tokens 4096  --update-freq 1 --no-progress-bar --log-interval 50 --keep-last-epochs 5 --max-epoch 55 \
  --init_method $init_method --init_state $init_state --init_scale $init_scale --init_scale_m0 $init_scale_m0 --use_warmup $use_warmup  \
  > nohups/exp_transformer-${alg}-${lr}-${eps}-${seed}_init${init_method}-${init_state}-${init_scale}-${init_scale_m0}_${use_warmup}.log 2>&1 &
done

init_method='grad-mean'
init_state='v'
export CUDA_VISIBLE_DEVICES=2;
for init_scale in 1 100
do
python -u  main.py --data ${DATA_PATH} --seed $seed --optimizer $alg --adam-betas '(0.9, 0.98)' --adam-eps $eps \
   --lr $lr --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
  --arch ${ARCH} --share-all-embeddings  --clip-norm 0.0  --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
  --max-tokens 4096  --update-freq 1 --no-progress-bar --log-interval 50 --keep-last-epochs 5 --max-epoch 55 \
  --init_method $init_method --init_state $init_state --init_scale $init_scale --init_scale_m0 $init_scale_m0 --use_warmup $use_warmup  \
  > nohups/exp_transformer-${alg}-${lr}-${eps}-${seed}_init${init_method}-${init_state}-${init_scale}-${init_scale_m0}_${use_warmup}.log 2>&1 &
done

init_method='grad-sq'
init_state='v'
export CUDA_VISIBLE_DEVICES=3;
for init_scale in 1 100
do
python -u  main.py --data ${DATA_PATH} --seed $seed --optimizer $alg --adam-betas '(0.9, 0.98)' --adam-eps $eps \
   --lr $lr --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
  --arch ${ARCH} --share-all-embeddings  --clip-norm 0.0  --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
  --max-tokens 4096  --update-freq 1 --no-progress-bar --log-interval 50 --keep-last-epochs 5 --max-epoch 55 \
  --init_method $init_method --init_state $init_state --init_scale $init_scale --init_scale_m0 $init_scale_m0 --use_warmup $use_warmup  \
  > nohups/exp_transformer-${alg}-${lr}-${eps}-${seed}_init${init_method}-${init_state}-${init_scale}-${init_scale_m0}_${use_warmup}.log 2>&1 &
done

init_method='grad-mean-var'
init_state='v'
export CUDA_VISIBLE_DEVICES=4;
for init_scale in 1 100
do
python -u  main.py --data ${DATA_PATH} --seed $seed --optimizer $alg --adam-betas '(0.9, 0.98)' --adam-eps $eps \
   --lr $lr --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
  --arch ${ARCH} --share-all-embeddings  --clip-norm 0.0  --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
  --max-tokens 4096  --update-freq 1 --no-progress-bar --log-interval 50 --keep-last-epochs 5 --max-epoch 55 \
  --init_method $init_method --init_state $init_state --init_scale $init_scale --init_scale_m0 $init_scale_m0 --use_warmup $use_warmup  \
  > nohups/exp_transformer-${alg}-${lr}-${eps}-${seed}_init${init_method}-${init_state}-${init_scale}-${init_scale_m0}_${use_warmup}.log 2>&1 &
done


init_method='grad-mean-random'
init_state='v'
export CUDA_VISIBLE_DEVICES=5;
for init_scale in 1 100
do
python -u  main.py --data ${DATA_PATH} --seed $seed --optimizer $alg --adam-betas '(0.9, 0.98)' --adam-eps $eps \
   --lr $lr --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
  --arch ${ARCH} --share-all-embeddings  --clip-norm 0.0  --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
  --max-tokens 4096  --update-freq 1 --no-progress-bar --log-interval 50 --keep-last-epochs 5 --max-epoch 55 \
  --init_method $init_method --init_state $init_state --init_scale $init_scale --init_scale_m0 $init_scale_m0 --use_warmup $use_warmup  \
  > nohups/exp_transformer-${alg}-${lr}-${eps}-${seed}_init${init_method}-${init_state}-${init_scale}-${init_scale_m0}_${use_warmup}.log 2>&1 &
done
