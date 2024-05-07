for i in {1..10}; do
    if [ ! -d "output/messy_room/old_room_25_with_mlp_triplet_loss_rgb_grads/point_cloud/iteration_80000" ]; then
        python train.py -s data/messy_room/old_room_25 -m output/messy_room/old_room_25_with_mlp_triplet_loss_rgb_grads --max_objects 12 --dataset messy_room --iteration 80000
    fi
    if [ ! -d "output/messy_room/old_room_50_with_mlp_triplet_loss_rgb_grads/point_cloud/iteration_80000" ]; then
        python train.py -s data/messy_room/old_room_50 -m output/messy_room/old_room_50_with_mlp_triplet_loss_rgb_grads --max_objects 12 --dataset messy_room --iteration 80000
    fi
    if [ ! -d "output/messy_room/old_room_100_with_mlp_triplet_loss_rgb_grads/point_cloud/iteration_80000" ]; then
        python train.py -s data/messy_room/old_room_100 -m output/messy_room/old_room_100_with_mlp_triplet_loss_rgb_grads --max_objects 12 --dataset messy_room --iteration 80000
    fi
    if [ ! -d "output/messy_room/old_room_500_with_mlp_triplet_loss_rgb_grads/point_cloud/iteration_80000" ]; then
        python train.py -s data/messy_room/old_room_500 -m output/messy_room/old_room_500_with_mlp_triplet_loss_rgb_grads --max_objects 12 --dataset messy_room --iteration 80000
    fi
done

list_dir=(15000 21000 30000 40000 50000 60000 80000)
replica_scenes=('old_room_25_with_mlp_triplet_loss_rgb_grads' 'old_room_50_with_mlp_triplet_loss_rgb_grads' 'old_room_100_with_mlp_triplet_loss_rgb_grads' 'old_room_500_with_mlp_triplet_loss_rgb_grads')

for j in "${replica_scenes[@]}"; do
    for i in "${list_dir[@]}"; do
        python render.py -m "output/messy_room/$j" --max_objects 12 --dataset messy_room --iteration "$i" --skip_train
    done
done


