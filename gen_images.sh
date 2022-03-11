dataset='cifar10'
arch='resnet34'

if [ ${dataset} -eq "cifar10" ];then
	epoch=2000
	bn_scale=10
fi

if [ ${dataset} -eq "cifar100" ];then
	epoch=4000
	bn_scale=3
fi

python NaturalInversion.py \
	--dataset $dataset \
	--arch $arch \
	--bs 256 \
	--iters_mi $epoch \
	--G_lr 0.001 \
	--D_lr 0.0005 \
	--A_lr 0.05 \
	--var_scale 6.0E-03 \
	--l2_scale 1.5E-05 \
	--r_feature_weight $bn_scale \
	--teacher_weight pretrained/cifar10_resnet34_9557.pt \
	--exp_name "$dataset"_"$arch"_paperparameters \
	--global_iter 0
