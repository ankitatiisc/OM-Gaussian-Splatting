for i in {1..10}; do
    if [ ! -d "output/scannet/2_grads/scene0354_00_with_mlp_cont_loss_2_grads/point_cloud/iteration_50000" ]; then
        python train.py -s /data/jaswanth/scannet/scene0354_00 -m output/scannet/2_grads/scene0354_00_with_mlp_cont_loss_2_grads --max_objects 12 --dataset scannet --iteration 50000
    fi
    if [ ! -d "output/scannet/2_grads/scene0389_00_with_mlp_cont_loss_2_grads/point_cloud/iteration_50000" ]; then
        python train.py -s /data/jaswanth/scannet/scene0389_00 -m output/scannet/2_grads/scene0389_00_with_mlp_cont_loss_2_grads --max_objects 12 --dataset scannet --iteration 50000
    fi
    if [ ! -d "output/scannet/2_grads/scene0423_02_with_mlp_cont_loss_2_grads/point_cloud/iteration_50000" ]; then
        python train.py -s /data/jaswanth/scannet/scene0423_02 -m output/scannet/2_grads/scene0423_02_with_mlp_cont_loss_2_grads --max_objects 12 --dataset scannet --iteration 50000
    fi
    if [ ! -d "output/scannet/2_grads/scene0427_00_with_mlp_cont_loss_2_grads/point_cloud/iteration_50000" ]; then
        python train.py -s /data/jaswanth/scannet/scene0427_00 -m output/scannet/2_grads/scene0427_00_with_mlp_cont_loss_2_grads --max_objects 12 --dataset scannet --iteration 50000
    fi
done

list_dir=(15000 21000 30000 40000 50000 80000)
scannet_scenes=('scene0221_01_with_cont_loss_2_grads' 'scene0050_02_with_cont_loss_2_grads' 'scene0144_01_with_cont_loss_2_grads' 'scene0300_01_with_cont_loss_2_grads' 'scene693_00_with_cont_loss_2_grads' 'scene616_00_with_cont_loss_2_grads')

for j in "${scannet_scenes[@]}"; do
    for i in "${list_dir[@]}"; do
        python render.py -m "output/scannet/2_grads/$j" --max_objects 12 --dataset scannet --iteration "$i" --skip_train
    done
done


