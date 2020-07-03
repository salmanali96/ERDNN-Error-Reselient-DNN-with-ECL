import torch
import torch.nn as nn
import numpy as np
import math
import random
from Activation import PReLU
import os
import itertools

#file = '/home/mlvcgpu/salman/MNIST/LUT1_' + str(bit_error) + '.npy'
'''
LUT = np.load('/home/mlvcgpu/salman/MNIST/LUT1_0.001.npy')
LUT = np.array((LUT))
'''


class ErrorCorrection():
    def LookUpTable(self, error_rate):
        path = './Error_Files/LUT1_' + str(error_rate) + '.npz'
        path = os.path.abspath(path)
        LUT = np.load(path)['LUT']
        return LUT

    def Check12(self, x):
        x1 = x.clone()
        max_n = torch.max(x)
        numofbits = self.countBits(int(max_n))
        x1 = x1 * 2 ** (31 - numofbits)
        shap = x.shape
        x1 = torch.reshape(x1, (-1, 1))
        x1 = x1.detach()
        x1 = x1.cpu()
        x1 = x1.numpy()
        x1 = x1.astype(int)
        x1 = x1.astype(float)
        x1 = torch.from_numpy(x1).float()
        x1 = x1/ (2 ** (31 - numofbits))
        x1 = torch.reshape(x1, (shap[0], shap[1], shap[2], shap[3]))
        x1 = x1.cuda()
        x.data = x1.data
        return x

    def countBits(self,number):
        try:
            return int((math.log(number) /
                        math.log(2)) + 1);
        except:
            return 1

    '''
    def BitError(self, x):
        x1 = x.clone()
        max_n = torch.max(x)
        numofbits = self.countBits(int(max_n))
        global rat

        x1 = x1 * 2 ** (31 - numofbits)

        shap = x.shape
        parameters = shap[0] * shap[1] * shap[2] * shap[3]
        x1 = torch.reshape(x1, (-1, 1))
        x1 = x1.detach()
        x1 = x1.cpu()
        x1 = x1.numpy()
        x1 = x1.astype(int)

        try:
            number = random.randint(0, 30000000 + 1 - parameters)
        except:
            global NOPS
            NOPS = NOPS + 1
            # print ('Too big tensor to operate. Problem occurred')
            return x

        lut = self.LUT[number: number + parameters]
        lut = lut.reshape((-1, 1))

        x1 = np.bitwise_xor(x1, lut)

        x1 = x1.astype(float)
        x1 = torch.from_numpy(x1).float()

        x1 = x1/ (2 ** (31 - numofbits))
        print (torch.max(x1))
        x1 = torch.reshape(x1, (shap[0], shap[1], shap[2], shap[3]))
        x1 = x1.cuda()
        x.data = x1.data
        return x
    '''

    def BitError(self, x):
        x1 = x.clone()
        max_n = torch.max(x)
        numofbits = self.countBits(int(max_n))
        global rat

        x1 = x1 * 2 ** (31 - numofbits)

        shap = x.shape
        parameters = shap[0] * shap[1] * shap[2] * shap[3]
        x1 = torch.reshape(x1, (-1, 1))
        #x1 = x1.detach()
        #x1 = x1.cpu()
        #x1 = x1.numpy()
        #x1 = x1.astype(int)
        x1 = x1.type(torch.int32)

        try:
            number = random.randint(0, 30000000 + 1 - parameters)
        except:
            # global NOPS
            # NOPS = NOPS + 1
            # print ('Too big tensor to operate. Problem occurred')
            return x

        lut = self.LUT[number: number + parameters]
        lut = lut.reshape((-1, 1))
        lut = torch.from_numpy(lut)
        lut = lut.cuda()
        x1 = torch.bitwise_xor(x1,lut)

        x1 = x1.type(torch.float)

        x1 = x1/ (2 ** (31 - numofbits))
        x1 = torch.reshape(x1, (shap[0], shap[1], shap[2], shap[3]))
        x1 = x1.cuda()
        x.data = x1.data
        return x

    def Tonumpy(self, f3):
        f3 = f3.detach()
        f3 = f3.cpu()
        f3 = f3.numpy()
        return f3

    def Precision(self, x1, numofbits):
        x1 = x1 * 2 ** (31 - numofbits)
        return x1

    def RestorePrecision(self, x1, numofbits):
        x1 = x1 / (2 ** (31 - numofbits))
        return x1

    def WeightCorruption(self, x, number):
        index = number
        weight = self.Conv_num(number)
        original_weight = weight.clone()
        max_n = torch.max(weight)
        numofbits = self.countBits(int(max_n))

        weight = weight * 2 ** (31 - numofbits)

        shap = weight.shape
        parameters = shap[0] * shap[1] * shap[2] * shap[3]
        x1 = weight
        x1 = torch.reshape(x1, (-1, 1))
        x1 = x1.detach()
        x1 = x1.cpu()
        x1 = x1.numpy()
        x1 = x1.astype(int)

        number = random.randint(0, 30000000 + 1 - parameters)
        lut = self.LUT[number: number + parameters]
        lut = lut.reshape((-1, 1))

        x1 = np.bitwise_xor(x1, lut)
        '''
        sign = np.sign(x1)
        check = sign * x1
        check = np.bitwise_and(check, 4294967040)
        x1 = check * sign
        '''
        x1 = x1.astype(float)

        x1 = torch.from_numpy(x1).float()
        x1 = x1 / (2 ** (31 - numofbits))

        x1 = torch.reshape(x1, (shap[0], shap[1], shap[2], shap[3]))
        x1 = x1.cuda()

        weight = x1

        self.WeightAssign(index,weight)

        x1 = self.BitError(x.clone())
        x = self.Convolution(x1.clone(), index)

        self.WeightAssign(index, original_weight)
        return x

    def WeightAssign(self, number, weight):
        if number == 1:
            #print (str(len(torch.nonzero(self.conv1.weight - weight))))
            weight = torch.nn.Parameter(weight)
            self.conv1.weight = weight
            #print (str(len(torch.nonzero(self.conv1.weight - weight))))
        elif number == 2:
            weight = torch.nn.Parameter(weight)
            self.conv2.weight = weight
        elif number == 3:
            weight = torch.nn.Parameter(weight)
            self.conv3.weight = weight
        elif number == 4:
            weight = torch.nn.Parameter(weight)
            self.conv4.weight = weight
        elif number == 5:
            weight = torch.nn.Parameter(weight)
            self.conv5.weight = weight
        elif number == 6:
            weight = torch.nn.Parameter(weight)
            self.conv6.weight = weight
        elif number == 7:
            weight = torch.nn.Parameter(weight)
            self.conv7.weight = weight
        elif number == 8:
            weight = torch.nn.Parameter(weight)
            self.conv8.weight = weight
        elif number == 9:
            weight = torch.nn.Parameter(weight)
            self.conv9.weight = weight
        elif number == 10:
            weight = torch.nn.Parameter(weight)
            self.conv10.weight = weight
        elif number == 11:
            weight = torch.nn.Parameter(weight)
            self.conv11.weight = weight
        elif number == 12:
            weight = torch.nn.Parameter(weight)
            self.conv12.weight = weight
        elif number == 13:
            weight = torch.nn.Parameter(weight)
            self.conv13.weight = weight

    '''
    def errorcorrection(self,x1,x2,x3, x4, x5, x6, x7):
        a = x1.clone()

        max_n = torch.max(x1)
        numofbits = self.countBits(int(max_n))
        x1 = self.Precision(x1, numofbits)
        x2 = self.Precision(x2, numofbits)
        x3 = self.Precision(x3, numofbits)
        x4 = self.Precision(x4, numofbits)
        x5 = self.Precision(x5, numofbits)
        x6 = self.Precision(x6, numofbits)
        x7 = self.Precision(x7, numofbits)

        x1 = self.Tonumpy(x1)
        x2 = self.Tonumpy(x2)
        x3 = self.Tonumpy(x3)
        x4 = self.Tonumpy(x4)
        x5 = self.Tonumpy(x5)
        x6 = self.Tonumpy(x6)
        x7 = self.Tonumpy(x7)

        x1 = x1.astype(int)
        x2 = x2.astype(int)
        x3 = x3.astype(int)
        x4 = x4.astype(int)
        x5 = x5.astype(int)
        x6 = x6.astype(int)
        x7 = x7.astype(int)

        Element = {}
        Element['A'] = x1
        Element['B'] = x2
        Element['C'] = x3
        Element['D'] = x4
        Element['E'] = x5
        Element['F'] = x6
        Element['G'] = x7

        List_results = []

        for subset in itertools.combinations(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 4):
            # print('Combination : ', subset)
            List_results.append(np.bitwise_and(Element[subset[0]], np.bitwise_and(Element[subset[1]],
                                                                                  np.bitwise_and(Element[subset[2]],
                                                                                                 Element[subset[3]]))))

        result = 0
        for i in range(0, len(List_results)):
            result = np.bitwise_or(result, List_results[i])

        x1 = result
        x1 = x1.astype(float)

        x1 = torch.from_numpy(x1).float()

        x1 = self.RestorePrecision(x1, numofbits)
        x1 = x1.cuda()

        a.data = x1.data
        return a
    '''

    def  errorcorrection(self,x1,x2,x3):
        a = x1.clone()

        max_n = torch.max(x1)
        #print (max_n)
        numofbits = self.countBits(int(max_n))
        x1 = self.Precision(x1,numofbits)
        x2 = self.Precision(x2, numofbits)
        x3 = self.Precision(x3, numofbits)

        x1 = x1.type(torch.int32)
        x2 = x2.type(torch.int32)
        x3 = x3.type(torch.int32)

        #a1 = self.ToPositive(x1.copy(),x2.copy())
        #a2 = self.ToPositive(x1.copy(),x3.copy())
        #a3 = self.ToPositive(x2.copy(),x3.copy())

        a1 = torch.bitwise_and(x1,x2)
        a2 = torch.bitwise_and(x1,x3)
        a3 = torch.bitwise_and(x2,x3)

        result = torch.bitwise_or(a1, a2)
        result = torch.bitwise_or(result, a3)

        x1 = result
        x1 = x1.type(torch.float)

        #x1 = torch.from_numpy(x1).float()
        #x2 = torch.from_numpy(x2).float()

        x1 = self.RestorePrecision(x1, numofbits)
        x1 = x1.cuda()

        a.data = x1.data
        return a

    '''
    def errorcorrection(self, x1, x2, x3, x4, x5):
        a = x1.clone()

        max_n = torch.max(x1)
        numofbits = self.countBits(int(max_n))
        x1 = self.Precision(x1,numofbits)
        x2 = self.Precision(x2, numofbits)
        x3 = self.Precision(x3, numofbits)
        x4 = self.Precision(x4, numofbits)
        x5 = self.Precision(x5, numofbits)

        x1 = self.Tonumpy(x1)
        x2 = self.Tonumpy(x2)
        x3 = self.Tonumpy(x3)
        x4 = self.Tonumpy(x4)
        x5 = self.Tonumpy(x5)

        x1 = x1.astype(int)
        x2 = x2.astype(int)
        x3 = x3.astype(int)
        x4 = x4.astype(int)
        x5 = x5.astype(int)

        arr = []
        arr.append(x1)
        arr.append(x2)
        arr.append(x3)
        arr.append(x4)
        arr.append(x5)

        AND = []
        for i in range(0, len(arr)):
            ch = []
            for j in range(i, len(arr)):
                if j + 1 >= len(arr):
                    break

                ch.append(np.bitwise_and(arr[i], arr[j + 1]))

            if len(ch) > 0:
                AND.append(ch)

        ch = []
        for i in range(0, len(AND)):
            for j in range(0, len(AND[i])):
                for k in range(j, len(AND[i])):
                    if k + 1 >= len(AND[i]):
                        break

                    ch.append(np.bitwise_and(AND[i][j], AND[i][k + 1]))

        result = 0
        for i in range(0, len(ch)):
            result = np.bitwise_or(result, ch[i])

        x1 = result
        x1 = x1.astype(float)

        x1 = torch.from_numpy(x1).float()

        x1 = self.RestorePrecision(x1, numofbits)
        x1 = x1.cuda()

        a.data = x1.data
        return a
    '''

    def Convolution(self, x, number):
        fn = getattr(self, 'conv' + str(number), None)
        x = fn(x)
        return x

    def Conv_num(self,number):
        weight = 'self.conv' + str(number) + '.weight'
        x = eval(weight)
        return x

    def errorcorrection1(self, x, number):
         xoriginal = self.Convolution(x.clone(), number)
          #return xoriginal

         x1 = xoriginal.clone()
         x1 = self.BitError(x1)
         return x1

         x2 = xoriginal.clone()
         x2 = self.BitError(x2)

         x3 = xoriginal.clone()
         x3 = self.BitError(x3)

         x = self.errorcorrection(x1, x2, x3)
         return x

    def ResErrorCorrection(self, x, number):
        xoriginal = self.ResConvl(x.clone(), number)
        #return xoriginal

        x1 = xoriginal.clone()
        x1 = self.BitError(x1)
       # return x1

        x2 = xoriginal.clone()
        x2 = self.BitError(x2)

        x3 = xoriginal.clone()
        x3 = self.BitError(x3)

        x = self.errorcorrection(x1, x2, x3)
        return x

    def ResConvl(self, x, number):
        fn = getattr(self, 'rcon' + str(number), None)
        x = fn(x)
        return x

    def Activation(self,Activation):
        if Activation == 'PRELU1':
            return PReLU.PRELU()
        elif Activation == 'PRELU2':
            return PReLU.PRELU2()
        elif Activation == 'PRELU3':
            return PReLU.PRELU3()
        elif Activation == 'PRELU4':
            return PReLU.PRELU4()
        elif Activation == 'PRELU5':
            return PReLU.PRELU5()
        elif Activation == 'PRELU6':
            return PReLU.PRELU6()
        elif Activation == 'RELU':
            return nn.ReLU()
