for i in {1..10}; do
    if [ ! -d "output/scannet/scene0494_00_with_mlp_3dcl_loss_rgb_grads/point_cloud/iteration_80000" ]; then
        python train.py -s data/scannet/scene0494_00 -m output/scannet/scene0494_00_with_mlp_3dcl_loss_rgb_grads --max_objects 12 --dataset scannet --iteration 80000
    fi
    if [ ! -d "output/scannet/scene0616_00_with_mlp_3dcl_loss_rgb_grads/point_cloud/iteration_80000" ]; then
        python train.py -s data/scannet/scene0616_00 -m output/scannet/scene0616_00_with_mlp_3dcl_loss_rgb_grads --max_objects 12 --dataset scannet --iteration 80000
    fi
    if [ ! -d "output/scannet/scene0693_00_with_mlp_3dcl_loss_rgb_grads/point_cloud/iteration_80000" ]; then
        python train.py -s data/scannet/scene0693_00 -m output/scannet/scene0693_00_with_mlp_3dcl_loss_rgb_grads --max_objects 12 --dataset scannet --iteration 80000
    fi
done

list_dir=(15000 21000 30000 40000 50000 80000)
scannet_scenes=('scene0494_00_with_mlp_3dcl_loss_rgb_grads' 'scene0616_00_with_mlp_3dcl_loss_rgb_grads' 'scene0693_00_with_mlp_3dcl_loss_rgb_grads')

for j in "${scannet_scenes[@]}"; do
    for i in "${list_dir[@]}"; do
        python render.py -m "output/scannet/$j" --max_objects 12 --dataset scannet --iteration "$i" --skip_train
    done
done


