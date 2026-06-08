import argparse


def get_args():
    parser = argparse.ArgumentParser("Hyperparameter Setting for main_code")

    parser.add_argument("--dino_port", type=int, default=12081)
    parser.add_argument("--blip_port", type=int, default=12082)
    parser.add_argument("--sam_port", type=int, default=12083)
    parser.add_argument("--yolo_port", type=int, default=12084)

    parser.add_argument("--card_select", type=int, default=3) # card:1 or 3
    parser.add_argument("--process_id", type=int, default=1) # 2-->4

    # parse arguments
    args = parser.parse_args()
    return args

args = get_args()


if(args.card_select==0): # 0
    args.dino_port = 12081
    args.blip_port = 12082
    args.sam_port = 12083
    args.yolo_port = 12084

# if(args.card_select==0): # 1
#     args.dino_port = 24001
#     args.blip_port = 24002
#     args.sam_port = 24003
#     args.yolo_port = 24004

elif(args.card_select==1):
    args.dino_port = 12181
    args.blip_port = 12182
    args.sam_port = 12183
    args.yolo_port = 12184

# elif(args.card_select==1):
#     args.dino_port = 12581
#     args.blip_port = 12582
#     args.sam_port = 12583
#     args.yolo_port = 12584

elif(args.card_select==2): # 2
    args.dino_port = 12281
    args.blip_port = 12282
    args.sam_port = 12283
    args.yolo_port = 12284

# elif(args.card_select==2): # 3
#     args.dino_port = 12681
#     args.blip_port = 12682
#     args.sam_port = 12683
#     args.yolo_port = 12684

elif(args.card_select==3): # 4
    args.dino_port = 12381
    args.blip_port = 12382
    args.sam_port = 12383
    args.yolo_port = 12384

# elif(args.card_select==3): # 5
#     args.dino_port = 32381
#     args.blip_port = 32382
#     args.sam_port = 32383
#     args.yolo_port = 32384


elif(args.card_select==4): # 6
    args.dino_port = 22481
    args.blip_port = 22482
    args.sam_port = 22483
    args.yolo_port = 22484

# elif(args.card_select==4): # 7
#     args.dino_port = 22881
#     args.blip_port = 22882
#     args.sam_port = 22883
#     args.yolo_port = 22884


# elif(args.card_select==4): # 13
#     args.dino_port = 12481
#     args.blip_port = 12482
#     args.sam_port = 12483
#     args.yolo_port = 12484 


# elif(args.card_select==5): # 8
#     args.dino_port = 12581
#     args.blip_port = 12582
#     args.sam_port = 12583
#     args.yolo_port = 12584

# elif(args.card_select==5): # 9
#     args.dino_port = 52581
#     args.blip_port = 52582
#     args.sam_port = 52583
#     args.yolo_port = 52584

elif(args.card_select==5): # 15
    args.dino_port = 52581
    args.blip_port = 52586
    args.sam_port = 52583
    args.yolo_port = 52584


# elif(args.card_select==6): # 10
#     args.dino_port = 62681
#     args.blip_port = 62682
#     args.sam_port = 62683
#     args.yolo_port = 62684

# elif(args.card_select==6): # 11
#     args.dino_port = 62681
#     args.blip_port = 62684
#     args.sam_port = 62683
#     args.yolo_port = 62684

elif(args.card_select==6): # 14
    args.dino_port = 62681
    args.blip_port = 62686
    args.sam_port = 62683
    args.yolo_port = 62684

elif(args.card_select==7): # 12
    args.dino_port = 12781
    args.blip_port = 12782
    args.sam_port = 12783
    args.yolo_port = 12784

# elif(args.card_select==7):
#     args.dino_port = 92781
#     args.blip_port = 27476
#     args.sam_port = 92783
#     args.yolo_port = 92784


# elif(args.card_select==7):
#     args.dino_port = 72781
#     args.blip_port = 72782
#     args.sam_port = 72783
#     args.yolo_port = 72784


