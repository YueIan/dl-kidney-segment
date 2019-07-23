from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import PIL
# from BagData import test_dataloader, train_dataloader
from BrainPackage.CNN.Dataset.JPGDataForSegment import JPGDataForSegment
from BrainPackage.CNN.Dataset.RAWDataForSegment import RAWDataForSegment
from BrainPackage.CNN.Model.FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from BrainPackage.CNN.Model.uNet import UNet
from BrainPackage.CNN.Model.PSPNet import PSPNet
from BrainPackage.DataPrepPocessing import RandomDataSelect
from BrainPackage.CNN.Loss.FocalLoss import FocalLoss
from BrainPackage.CNN.Evaluate.KidneyEvaluate import acc1an2
from BrainPackage.CNN.Evaluate.KidneyEvaluate import acc1an2forOneSlicein3D
from BrainPackage.CNN.Evaluate.KidneyEvaluate import acc1an2in3D
from BrainPackage.CNN.Evaluate.KidneyEvaluate import accall
from BrainPackage.CNN.Evaluate.KidneyEvaluate import AccStruct
# import visdom
class SegmentTestInfo:
    train_image_root =''
    test_image_root =''
    test_mask_root =''
    train_mask_root =''
    batch_size =4
    checkpoint_name =''
    echo_number = 10
    data_size = 160
    saved_train_data_name =''
    saved_test_data_name =''
    stamp = ''
    thread_number = 0
    colordim = 3
    imagetype = 'left'
    output_path = r'c:\Temp'
class ParcelFCNFocusLossTest:
    '''This is CNN test case  '''
    def __init__(self, test_info, is_load_check_point = False, is_random_select = 'Load', data_percent = [0.8,0.2], net_type= 'FCN', segment_number = 2, weight = [0,0,1,0], focus_weight =[0.1,0.3,0.6], is_only_test = False):
        self.train_image_root = test_info.train_image_root
        self.test_image_root = test_info.test_image_root
        self.train_mask_root = test_info.train_mask_root
        self.test_mask_root = test_info.test_mask_root
        self.batch_size = test_info.batch_size
        self.is_load_check_point = is_load_check_point
        self.checkpoint_name = test_info.checkpoint_name
        self.echo_number = test_info.echo_number
        self.is_random_select = is_random_select
        self.data_percent = data_percent
        self.net_type = net_type
        self.data_size = test_info.data_size
        self.saved_train_data_name = test_info.saved_train_data_name 
        self.saved_test_data_name = test_info.saved_test_data_name
        self.stamp = test_info.stamp
        self.segment_number = segment_number
        self.weight = weight
        self.focus_weight = focus_weight
        self.is_only_test = is_only_test
        self.thread_number = test_info.thread_number
        self.output_png = True
        self.colordim = test_info.colordim
        self.imagetype = test_info.imagetype
        self.output_folder = test_info.output_path
        if is_only_test == True:
            self.echo_number =1; 
    def PerformTestCase(self):
        self.PrepareTestCase(show_vgg_params=False)
        self.RunTestCase(epo_num=self.echo_number)
    def PrepareTestCase(self,show_vgg_params=False):
        #Prepare training data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        if self.colordim == 3: # png
            self.train_data = JPGDataForSegment(self.train_image_root, self.train_mask_root, data_size = self.data_size, segment_number = self.segment_number, weight = self.weight, is_need_onehot = False)
            self.test_data = JPGDataForSegment(self.test_image_root, self.test_mask_root,data_size = self.data_size, segment_number = self.segment_number, weight = self.weight, is_need_onehot = False)
        else: #raw
            self.train_data = RAWDataForSegment(self.train_image_root, self.train_mask_root, data_size = self.data_size, segment_number = self.segment_number, is_need_onehot = False)
            self.test_data = RAWDataForSegment(self.test_image_root, self.test_mask_root,data_size = self.data_size, segment_number = self.segment_number, is_need_onehot = False)
            self.colordim = 3
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.thread_number, drop_last = True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=False, num_workers=0)
        if self.is_random_select == 'Random':
            datalist= RandomDataSelect.GetRandomFileListByPercentListForSegment(self.train_image_root, self.data_percent)  
            self.train_data.loadDataFromExistedFileList(datalist[0])  
            self.test_data.loadDataFromExistedFileList(datalist[1])
            self.train_data.SaveDataAndLabelToTxt('stamp{}_traindata.txt'.format(self.stamp))
            self.test_data.SaveDataAndLabelToTxt('stamp{}_testdata.txt'.format(self.stamp))
        elif self.is_random_select == 'Load':
            self.train_data.LoadDataAndLabelFromtxt(self.saved_train_data_name)
            self.test_data.LoadDataAndLabelFromtxt(self.saved_test_data_name)
            self.train_data.SaveDataAndLabelToTxt('stamp{}_traindata_l.txt'.format(self.stamp))
            self.test_data.SaveDataAndLabelToTxt('stamp{}_testdata_l.txt'.format(self.stamp))

        if self.net_type == 'FCN':
            vgg_model = VGGNet(pretrained= True,requires_grad=True, show_params=show_vgg_params)
            self.fcn_model = FCNs(pretrained_net=vgg_model, n_class= self.segment_number, is_need_sigmoid = False, colordim = self.colordim)
        elif self.net_type == 'uNet':
            self.fcn_model = UNet(colordim = self.colordim, class_type = self.segment_number, is_need_sigmoid = False)
        else:
            self.fcn_model = PSPNet(num_classes = self.segment_number, pretrained=True, is_need_sigmoid = False, colordim = self.colordim)
        self.fcn_model = self.fcn_model.to(self.device)
        # self.criterion = nn.BCELoss().to(self.device)
        alpha_map = np.zeros((self.batch_size, self.segment_number, self.data_size, self.data_size), dtype = np.float32)
        alpha_map_test = np.zeros((1, self.segment_number, self.data_size, self.data_size), dtype = np.float32)
        for i in range(self.segment_number):
            alpha_map[:,i,:,:] = self.focus_weight[i]
            alpha_map_test[:,i,:,:] = self.focus_weight[i]
        alpha_map = torch.FloatTensor(alpha_map)
        alpha_map = alpha_map.to(self.device)
        alpha_map_test = torch.FloatTensor(alpha_map_test)
        alpha_map_test = alpha_map_test.to(self.device)
        self.criterion = FocalLoss(2, alpha_map)
        self.criterion_test = FocalLoss(2, alpha_map_test)
        self.optimizer = optim.SGD(self.fcn_model.parameters(), lr=0.02, momentum=0.7)
        if self.is_load_check_point == True:
            check_point_type = self.checkpoint_name.split(".")[-1]
            if check_point_type == 'pt':
                checkpoint = torch.load(self.checkpoint_name).state_dict()
                self.fcn_model.load_state_dict(checkpoint)
            else:
                self.fcn_model.load_state_dict(torch.load(self.checkpoint_name))
        self.test_ressult_number = self.test_data.sortresult()
        # test 
    def RunTestCase(self,epo_num=7):
        fp = open('stamp_{}_{}.log'.format(self.stamp, self.net_type), 'w')
        # vis = visdom.Visdom()

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
        # fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
        # fcn_model = fcn_model.to(device)
        # criterion = nn.BCELoss().to(device)
        # optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)
        zoom_factor = 255/(self.segment_number - 1);

        all_train_iter_loss = []
        all_test_iter_loss = []

        # start timing
        prev_time = datetime.now()
        for epo in range(epo_num):
            train_loss = 0
            if self.is_only_test == False:
                self.fcn_model.train()
                for index, (bag, bag_msk, raw_image, filename) in enumerate(self.train_loader):
                    # bag.shape is torch.Size([4, 3, 160, 160])
                    # bag_msk.shape is torch.Size([4, 2, 160, 160])
                    if np.mod(index, 200) == 0:
                        print('test process[%d-%d]' % (epo, index))
                    bag = bag.to(self.device)
                    bag_msk = bag_msk.to(self.device)

                    self.optimizer.zero_grad()
                    output = self.fcn_model(bag)
                    # output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                    # output_nb = output[0,0:80,:].cpu().detach().numpy().copy()
                    loss = self.criterion(output, bag_msk)
                    loss.backward()
                    iter_loss = loss.item()
                    all_train_iter_loss.append(iter_loss)
                    train_loss += iter_loss
                    self.optimizer.step()

                #     output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
                #     output_np = np.argmin(output_np, axis=1)
                #     bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
                #     bag_msk_np = np.argmin(bag_msk_np, axis=1)

                    # if np.mod(index, 15) == 0:
                        # print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(self.train_dataloader), iter_loss))
                        # vis.close()
                        # vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction')) 
                        # vis.images(bag_msk_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                        # vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))

                    # plt.subplot(1, 2, 1) 
                    # plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
                    # plt.subplot(1, 2, 2) 
                    # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
                    # plt.pause(0.5)

            
            test_loss = 0
            
            acc_tum_sum = []
            acc_kidney_sum =[]
            acc_for_sum =[]
            acc_tum_tt_sum =[]
            acc_tum_tf_sum=[]
            acc_tum_ft_sum=[]
            acc_kidney_tt_sum = []
            acc_kidney_tf_sum = []
            acc_kidney_ft_sum = []
            min_index =[0]
            max_index=[]
            center_index = []
            acc_result = AccStruct
            test_result_index = 0
            self.fcn_model.eval()
            with torch.no_grad():
                for index, (bag, bag_msk, raw_image,filename) in enumerate(self.test_loader):

                    bag = bag.to(self.device)
                    bag_msk = bag_msk.to(self.device)

                    self.optimizer.zero_grad()
                    output = self.fcn_model(bag)
                    # output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                    #bag_msk.shape is torch.Size([4, 160, 160])
                    loss = self.criterion_test(output, bag_msk)
                    iter_loss = loss.item()
                    all_test_iter_loss.append(iter_loss)
                    test_loss += iter_loss

                    output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
                    output_np = np.argmax(output_np, axis=1)
                    bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
                    # bag_msk_np = np.argmax(bag_msk_np, axis=1)
            
                    # if np.mod(index, 15) == 0:
                        # print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(self.test_dataloader), iter_loss))
                        # print(r'Testing... Open http://localhost:8097/ to see test result.')
                        # vis.close()
                        # vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction')) 
                        # vis.images(bag_msk_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                        # vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))

                    # plt.subplot(1, 2, 1) 
                    # plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
                    # plt.subplot(1, 2, 2) 
                    # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
                    # plt.pause(0.5)
                    if np.mod(epo, 1    ) == 0:
                        acc1an2forOneSlicein3D(bag_msk_np,output_np,acc_result,index)
                        if index == self.test_ressult_number[test_result_index]:
                            test_result_index +=1
                            # acc1, acc2, acc_tum = acc1an2in3D(acc_result)
                            # acc1_sum.append(acc1)
                            # acc2_sum.append(acc2)
                            acc_tum_tt_sum.append(acc_result.acc_2_tt)
                            acc_tum_tf_sum.append(acc_result.acc_2_tf)
                            acc_tum_ft_sum.append(acc_result.acc_2_ft)
                            acc_kidney_tt_sum.append(acc_result.acc1_1_tt)
                            acc_kidney_tf_sum.append(acc_result.acc1_1_tf)
                            acc_kidney_ft_sum.append(acc_result.acc1_1_ft)
                            center_index.append(acc_result.center_index)
                            acc_tum,acc_kidney,acc_fore = accall(acc_result)
                            acc_tum_sum.append(acc_tum)
                            acc_kidney_sum.append(acc_kidney)
                            acc_for_sum.append(acc_fore)
                            max_index.append(index)
                            min_index.append(index+1)

                    if epo == epo_num - 1 and self.output_png:
                        output_np = np.uint8(output_np * zoom_factor)
                        output_np = np.transpose(output_np, (1,2,0)) 
                        output_np = np.reshape(output_np, [output_np.shape[0], output_np.shape[1]])    
                        zero_np = np.zeros_like(output_np)
                        output_np = PIL.Image.fromarray(output_np)  
                        zero_np = PIL.Image.fromarray(zero_np)  
                        raw_image = raw_image.cpu().detach().numpy().copy() 
                        raw_image = np.reshape(raw_image, [raw_image.shape[1], raw_image.shape[2],raw_image.shape[3]])   
                        raw_image =  PIL.Image.fromarray(raw_image) 
                        r,g,b = raw_image.split()
                        
                        r = PIL.Image.blend(r, output_np, 0.5)
                        g = PIL.Image.blend(g, zero_np, 0.5)
                        b = PIL.Image.blend(b, zero_np, 0.5)
                        # result_name = r'D:\hanbing\resultdemo\result2\stamp_{}_result_{}_lung_{}.png'.format(self.stamp,self.net_type,index)
                        # label_name = r'D:\hanbing\resultdemo\label2\stamp_{}_target_{}_lung_{}.png'.format(self.stamp,self.net_type,index)
                        # blending_name = r'D:\hanbing\resultdemo\blending2\stamp_{}_blending_result_{}_lung_{}.png'.format(self.stamp, self.net_type, index)
                        # if test_result_index < len(self.test_ressult_number)//3:
                        #     result_name = r'D:\hanbing\result2\result\stamp_{}_result_{}_lung_{}.png'.format(self.stamp, self.net_type, index)
                        #     label_name = r'D:\hanbing\result2\label\stamp_{}_target_{}_lung_{}.png'.format(self.stamp, self.net_type, index)
                        #     blending_name =r'D:\hanbing\result2\blending\stamp_{}_blending_result_{}_lung_{}.png'.format(self.stamp, self.net_type, index)
                        # elif test_result_index < len(self.test_ressult_number) * 2//3:
                        #     result_name = r'D:\hanbing\result3\result\stamp_{}_result_{}_lung_{}.png'.format(self.stamp,self.net_type,index)
                        #     label_name = r'D:\hanbing\result3\label\stamp_{}_target_{}_lung_{}.png'.format(self.stamp,self.net_type,index)
                        #     blending_name = r'D:\hanbing\result3\blending\stamp_{}_blending_result_{}_lung_{}.png'.format(self.stamp, self.net_type, index)
                        # else:
                        #     result_name = r'D:\hanbing\result4\result\stamp_{}_result_{}_lung_{}.png'.format(self.stamp,self.net_type,index)
                        #     label_name = r'D:\hanbing\result4\label\stamp_{}_target_{}_lung_{}.png'.format(self.stamp, self.net_type,index)
                        #     blending_name = r'D:\hanbing\result4\blending\stamp_{}_blending_result_{}_lung_{}.png'.format(self.stamp, self.net_type, index)
                        result_name = os.path.join(self.output_folder, r'result',self.imagetype, filename[0])
                        label_name = os.path.join(self.output_folder, r'label',self.imagetype, filename[0])
                        # blending_name = os.path.join(r'D:\hanbing\resultdemo\blending4', self.imagetype,filename[0])
                        if not os.path.exists(self.output_folder):
                            os.mkdir(self.output_folder)
                        if not os.path.exists(os.path.join(self.output_folder, r'result')):
                            os.mkdir(os.path.join(self.output_folder, r'result'))
                        if not os.path.exists(os.path.join(self.output_folder, r'label')):
                            os.mkdir(os.path.join(self.output_folder, r'label'))
                        if not os.path.exists(os.path.join(self.output_folder, r'result',self.imagetype)):
                            os.mkdir(os.path.join(self.output_folder, r'result',self.imagetype))
                        if not os.path.exists(os.path.join(self.output_folder, r'label',self.imagetype)):
                            os.mkdir(os.path.join(self.output_folder, r'label',self.imagetype))
                        output_np.save(open(result_name, 'wb'))
                        output_np = PIL.Image.merge("RGB",[r,g,b])
                        # output_np.save(open(blending_name, 'wb'))
                        # raw_image.save(open(blending_name, 'wb'))
                        bag_msk_np = np.uint8(bag_msk_np * zoom_factor)
                        bag_msk_np = np.transpose(bag_msk_np, (1,2,0)) 
                        bag_msk_np = np.reshape(bag_msk_np, [bag_msk_np.shape[0], bag_msk_np.shape[1]])       
                        bag_msk_np = PIL.Image.fromarray(bag_msk_np)
                        bag_msk_np.save(open(label_name, 'wb'))
                        # bag_msk_np = PIL.Image.blend(bag_msk_np, raw_image, 0.5)
                        # bag_msk_np.save(open('stamp_{}_blending_target_{}_lung_{}.png'.format(self.stamp, self.net_type, index), 'wb'))
                           
                        # PIL.Image.fromarray(output_np).resize((800, 800)).save(open('result2_{}_bag_{}.png'.format(self.net_type, index), 'wb'))
                        # evaluate

            
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            prev_time = cur_time

            print('epoch train loss = %f, epoch test loss = %f, %s'
                    %(train_loss/len(self.train_loader), test_loss/len(self.test_loader), time_str))
            fp.write('train loss =' + str(train_loss/len(self.train_loader)) + ',test loss = ' + str(test_loss/len(self.test_loader)) + '\n' )

            if np.mod(epo, 1) == 0:
                torch.save(self.fcn_model, 'stamp_{}_{}_model_{}.pt'.format(self.stamp, self.net_type, epo))
                torch.save(self.fcn_model.state_dict(), 'stamp_{}_{}_model_{}.plt'.format(self.stamp, self.net_type, epo))
                print('saving stamp_{}_{}_model_{}.pt'.format(self.stamp, self.net_type, epo))
                average_acckidney = 0
                average_accforeground = 0
                average_acctum = 0
                for index in range(len(self.test_ressult_number)):
                    print('test acckidney[%d] = %f, test accForeground[%d] = %f, test acc_tum[%d] = %f, center_slice = %d[%d-%d]' %(index, acc_kidney_sum[index], index, acc_for_sum[index], index, acc_tum_sum[index], center_index[index], min_index[index],max_index[index]))
                    average_acckidney+= acc_kidney_sum[index]
                    average_accforeground+= acc_for_sum[index]
                    average_acctum+= acc_tum_sum[index]
                    fp.write('acckidney' + str(index) + '=' + str(acc_kidney_sum[index]) + ',accForeground' + str(index) + '=' + str(acc_for_sum[index]) + ',acc_tum' + str(index) +'=' + str(acc_tum_sum[index])+ '\n' )
                for index in range(len(self.test_ressult_number)):
                    fp.write('center_index' + str(index) +'='+str(center_index[index]) +'\n' )
                for index in range(len(self.test_ressult_number)):
                    fp.write('acctum_tt' + str(index) + '=' + str(acc_tum_tt_sum[index]) + ',acctum_tf' + str(index) + '=' + str(acc_tum_tf_sum[index]) + ',acctum_ft' + str(index) +'=' + str(acc_tum_ft_sum[index])+ '\n' )
                    print('test acctum_tt[%d] = %f, test acctum_tf[%d] = %f, test acctum_ft[%d] = %f' %(index, acc_tum_tt_sum[index], index, acc_tum_tf_sum[index], index, acc_tum_ft_sum[index]))
                for index in range(len(self.test_ressult_number)):
                    fp.write('acckidney_tt' + str(index) + '=' + str(acc_kidney_tt_sum[index]) + ',acckidney_tf' + str(index) + '=' + str(acc_kidney_tf_sum[index]) + ',acckidney_ft' + str(index) +'=' + str(acc_kidney_ft_sum[index])+ '\n' )
                    print('test acckidney_tt[%d] = %f, test acckidney_tf[%d] = %f, test acckidney_ft[%d] = %f' %(index, acc_kidney_tt_sum[index], index, acc_kidney_tf_sum[index], index, acc_kidney_ft_sum[index]))
                average_acckidney /= len(acc_tum_sum)
                average_accforeground /= len(acc_tum_sum)
                average_acctum /= len(acc_tum_sum)
                print('test acckidney = %f, test foreground= %f, test acc_tum= %f' %(average_acckidney,average_accforeground,average_acctum))
                fp.write('average_acckidney' + '=' + str(average_acckidney) + ',average_accforeground' + '=' + str(average_accforeground) + ',average_acc_tum' +'=' + str(average_acctum)+ '\n' )
        fp.close()    
                


if __name__ == "__main__":

    train(epo_num=100, show_vgg_params=False)
