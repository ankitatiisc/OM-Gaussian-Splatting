# python render.py -s data/replica/office_0 -m output/replica/office_0_with_mlp_triplet_loss_3d_los --max_objects 12 --dataset replica
# python render.py -s data/replica/office_2 -m output/replica/office_2_with_mlp_triplet_loss_3d_los --max_objects 12 --dataset replica
# python render.py -s data/replica/office_3 -m output/replica/office_3_with_mlp_triplet_loss_3d_los --max_objects 12 --dataset replica
# python render.py -s data/replica/office_4 -m output/replica/office_4_with_mlp_triplet_loss_3d_los --max_objects 12 --dataset replica

list_dir=(7000 15000 21000 30000 40000 50000)
replica_scenes=('office_0_with_mlp_triplet_loss_3d_los_2_grads' 'office_2_with_mlp_triplet_loss_3d_los_2_grads' 'office_3_with_mlp_triplet_loss_3d_los_2_grads' 'office_4_with_mlp_triplet_loss_3d_los_2_grads' 'room_1_with_mlp_triplet_loss_3d_los_2_grads' 'room_0_with_mlp_triplet_loss_3d_los_2_grads' 'room_2_with_mlp_triplet_loss_3d_los_2_grads')

for j in "${replica_scenes[@]}"; do
    for i in "${list_dir[@]}"; do
        python render.py -m "output/replica/$j" --max_objects 12 --dataset replica --iteration "$i" --skip_train
    done
done