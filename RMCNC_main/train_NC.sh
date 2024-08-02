# # 0-Scene15       
data=0
tau=1  
gpu=1
q=0.5 
lr=0.001
epoch=100


# 1-Caltech101   
# data=1
# tau=1 
# gpu=1
# q=2   
# lr=0.001   
# epoch=100



# # 3-NoisyMNIST       
# data=3
# tau=0.5 
# gpu=1
# q=0.5
# lr=0.001
# epoch=100



# # 5-DeepAnimal        
# data=5
# tau=0.5  
# gpu=1
# q=2               
# lr=0.0002
# epoch=100



# # 7-wiki_2_view       
# data=7
# tau=2
# gpu=1
# q=0.5
# lr=0.001
# epoch=100



# 9-nuswide_deep_2_view       
# data=9
# tau=1 
# gpu=1
# q=0.5
# lr=0.001
# epoch=100

# 10-xmedia_deep_2_view       
# data=10
# tau=0.5
# gpu=0
# q=0.5
# lr=0.001 
# epoch=100

# 11-rmb_2_view       
# data=11
# tau=1
# gpu=1
# q=2
# lr=0.001
# epoch=100


for align_rate in "0.8" "0.6" "0.4" "0.2"   
do 
    echo "===================================================================================="
    echo "================================alignment rate:" $align_rate "================================"
    echo "===================================================================================="
    for i in "1" "2" "3" "4" "5"
    do
        python run_RMCNC.py --data $data --tau $tau --gpu $gpu -ap $align_rate --NC --method pa --q $q -lr $lr --e $epoch -li $epoch
    done
done

