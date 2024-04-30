for i in {1..10}; do
    if [ ! -d "output/replica/room_2_with_mlp_triplet_loss_3d_los_2_grads/point_cloud/iteration_50000" ]; then
        python train.py -s data/replica/room_2 -m output/replica/room_2_with_mlp_triplet_loss_3d_los_2_grads --max_objects 12 --dataset replica --iteration 50000
    fi
    if [ ! -d "output/replica/office_4_with_mlp_triplet_loss_3d_los_2_grads/point_cloud/iteration_50000" ]; then
        python train.py -s data/replica/office_4 -m output/replica/office_4_with_mlp_triplet_loss_3d_los_2_grads --max_objects 12 --dataset replica --iteration 50000
    fi
done

list_dir=(15000 21000 30000 40000 50000)
replica_scenes=('office_4_with_mlp_triplet_loss_3d_los_2_grads' 'room_2_with_mlp_triplet_loss_3d_los_2_grads')

for j in "${replica_scenes[@]}"; do
    for i in "${list_dir[@]}"; do
        python render.py -m "output/replica/$j" --max_objects 12 --dataset replica --iteration "$i" --skip_train
    done
done

