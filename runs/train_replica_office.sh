for i in {1..10}; do
    if [ ! -d "output/replica/office_0_with_mlp_triplet_loss_rgb_grads/point_cloud/iteration_80000" ]; then
        python train.py -s data/replica/office_0 -m output/replica/office_0_with_mlp_triplet_loss_rgb_grads --max_objects 12 --dataset replica --iteration 80000
    fi
    if [ ! -d "output/replica/office_3_with_mlp_triplet_loss_rgb_grads/point_cloud/iteration_80000" ]; then
        python train.py -s data/replica/office_3 -m output/replica/office_3_with_mlp_triplet_loss_rgb_grads --max_objects 12 --dataset replica --iteration 80000
    fi
    if [ ! -d "output/replica/office_2_with_mlp_triplet_loss_rgb_grads/point_cloud/iteration_80000" ]; then
        python train.py -s data/replica/office_2 -m output/replica/office_2_with_mlp_triplet_loss_rgb_grads --max_objects 12 --dataset replica --iteration 80000
    fi
    if [ ! -d "output/replica/office_4_with_mlp_triplet_loss_rgb_grads/point_cloud/iteration_80000" ]; then
        python train.py -s data/replica/office_4 -m output/replica/office_4_with_mlp_triplet_loss_rgb_grads --max_objects 12 --dataset replica --iteration 80000
    fi
done

list_dir=(15000 21000 30000 40000 50000 80000)
replica_scenes=('office_0_with_mlp_triplet_loss_rgb_grads' 'office_3_with_mlp_triplet_loss_rgb_grads' 'office_2_with_mlp_triplet_loss_rgb_grads' 'office_4_with_mlp_triplet_loss_rgb_grads')

for j in "${replica_scenes[@]}"; do
    for i in "${list_dir[@]}"; do
        python render.py -m "output/replica/$j" --max_objects 12 --dataset replica --iteration "$i" --skip_train
    done
done


