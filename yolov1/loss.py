import torch
#trainData(7,7,30)
def GetLoss(trainData,testData):
    trainData=trainData.view(-1,7,7,30)
    testData=testData.view(-1,7,7,30)
    hasObj=trainData[:,:,:,4]==1.0
    noObj=trainData[:,:,:,4]==0
    tr_obj=trainData[hasObj]
    test_obj=trainData[hasObj]
    iou1=IOU(tr_obj[:,:4],test_obj[:,:4])
    iou2=IOU(tr_obj[:,:4],test_obj[:,5:9])
    MaxIou=iou2>iou1
    test_obj[MaxIou][:,:4]=test_obj[MaxIou][:,5:9]
    MaxIou=torch.max(iou1,iou2)
    #边框中心点误差
    cxy_Miss=(tr_obj[:,0]-test_obj[:,0])**2+(tr_obj[:,1]-test_obj[:,1])**2
    #print('边框中心点误差')
    #print(cxy_Miss)
    #边框宽度高度误差
    wh_Miss=(tr_obj[:,2]-test_obj[:,2])**2+(tr_obj[:,3]-test_obj[:,3])**2
    #print('边框宽度高度误差')
    #print(wh_Miss)
    #置信度误差
    c_miss=(tr_obj[:,4]-MaxIou)**2
    #print('置信度误差')
    #print(c_miss)
    #无对象置信度误差
    tr_noobj=trainData[noObj]
    test_noobj=testData[noObj]
    iou3=IOU(tr_noobj[:,:4],test_noobj[:,:4])
    iou4=IOU(tr_noobj[:,:4],test_noobj[:,5:9])
    minIou=torch.min(iou3,iou4)
    #print(minIou)
    cno_miss=(tr_noobj[:,4]-minIou)**2
    #print('无对象置信度误差')
    #print(cno_miss)
    #对象分类误差
    p_miss=(tr_obj[:,10:]-test_obj[:,10:])**2
    #print('对象分类误差')
    #print(p_miss)
    sum=5.0*torch.sum(cxy_Miss)+5.0*torch.sum(wh_Miss)\
        +torch.sum(c_miss)+0.5*torch.sum(cno_miss)+torch.sum(p_miss)
    #print(sum)
    return sum
def IOU(box1,box2):
    s1=box1[:,2]*box1[:,3]
    s2=box2[:,2]*box2[:,3]
    b1_x1=box1[:,0]-box1[:,2]/2
    b1_y1=box1[:,1]-box1[:,3]/2
    b1_x2=box1[:,0]+box1[:,2]/2
    b1_y2=box1[:,1]+box1[:,3]/2

    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    #计算交集
    xmin=torch.max(b1_x1,b2_x1)
    ymin=torch.max(b1_y1,b2_y1)
    xmax=torch.min(b1_x2,b2_x2)
    ymax=torch.min(b1_y2,b2_y2)
    w=torch.max(torch.zeros(xmax.shape),xmax-xmin)
    h=torch.max(torch.zeros(xmax.shape),ymax-ymin)
    s=w*h
    return s/(s1+s2-s)
