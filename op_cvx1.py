import numpy as np
import random
import scipy.integrate as integrate
from cvxopt  import solvers, matrix 
import matplotlib.pyplot as plt


'''
车辆编号： 0—ego   1—原车道后车  2—原车道前车  3—目标车道后车  4—目标车道前车
--------------------------------------------------------------------------------
                   3                                                                      4
--------------------------------------------------------------------------------
                               1                       0                            2
--------------------------------------------------------------------------------
'''
'''
说明：简化版本
1、自变量不考虑jerk
2、等式约束仅考虑速度和加速度
3、不等式约束考虑防碰撞约束、后车不超过前车约束
4、路径采用三次曲线
'''

def cca(v1, v2, x12):
    a1 = 0.0343 * (x12 - 30) + 0.948 * (v2 - v1) #Pariota跟驰模型
    return a1

#生成不同的车队环境
def scenario_generation():
    #存储ego车辆及周边车辆状态信息
    sgenMatrix = np.zeros((NumGen, 5*3))
    #以下生成变量为纵向要素
    for m in range(0, NumGen):
        sg = 0
        v3v4dis = 0
        #随机生成初始速度和间距，循环直到找到可以换道的scenario
        while sg == 0:
            veh0v0 = random.uniform(11, 21)      #ego
            veh1v0 = random.uniform(0.9, 1.1) * veh0v0   #后车
            veh2v0 = random.uniform(0.9, 1.1) * veh0v0   #前车
            veh3v0 = random.uniform(1.2, 1.6) * veh0v0   #目标后车
            veh4v0 = random.uniform(0.9, 1.1) * veh3v0   #目标前车
            
            v3v4dis = random.uniform(0.9, 1.1)*(1000/(150 - (150/80) * (veh3v0*3600/1000)))   #应该是随机创建了一个间距
            if v3v4dis > 15 and (veh0v0 < 22 and veh1v0 < 22 and veh2v0 < 22) and (veh3v0 < 22 and veh4v0 < 22):
                sg = 1
        
        #随机生成纵向位置
        veh3x0 = 100
        veh4x0 = veh3x0 + v3v4dis 
        veh0x0 = veh3x0 + random.uniform(5, v3v4dis - 5)
        v0v2dis = random.uniform(0.9, 1.1)*(1000/(150 - (150/80) * (veh0v0*3600/1000)))   #km/h
        v1v0dis = random.uniform(0.9, 1.1)*(1000/(150 - (150/80) * (veh1v0*3600/1000)))  
        veh2x0 = veh0x0 + v0v2dis
        veh1x0 = veh0x0 - v1v0dis
        
        #随机生成纵向加速度
        veh4a0 = random.uniform(-1, 1)
        veh2a0 = random.uniform(-1, 1)
        veh3a0 = cca(veh3v0, veh4v0, v3v4dis)
        veh0a0 = cca(veh0v0, veh2v0, v0v2dis)
        veh1a0 = cca(veh1v0, veh0v0, v1v0dis)
        
        #存入numpy array
        sgenMatrix[m, 0] = veh0x0
        sgenMatrix[m, 1] = veh0v0
        sgenMatrix[m, 2] = veh0a0
        
        sgenMatrix[m, 3] = veh1x0
        sgenMatrix[m, 4] = veh1v0
        sgenMatrix[m, 5] = veh1a0
        
        sgenMatrix[m, 6] = veh2x0
        sgenMatrix[m, 7] = veh2v0
        sgenMatrix[m, 8] = veh2a0

        sgenMatrix[m, 9] = veh3x0
        sgenMatrix[m, 10] = veh3v0
        sgenMatrix[m, 11] = veh3a0
        
        sgenMatrix[m, 12] = veh4x0
        sgenMatrix[m, 13] = veh4v0
        sgenMatrix[m, 14] = veh4a0

    # print(sgenMatrix)
    return sgenMatrix

#获取周边信息，设定候选终点，对每个候选方案进行QP模型求解
def qp_execution(sgenMatrix):
    vehlen = 4.8       #车长
    vehwid = 1.8      #车宽
    
    yxmain = lambda x: 0   #原车道中心线y值   y = f(x)
    dydxmain = lambda x: 0    #导函数
    
    yxtarget = lambda x: 3.75   #目标车道中心线y值
    dydxtarget  =lambda x:0
    
    for m in range(NumGen):
        #获取x,y,v,a信息
        veh0x0 = sgenMatrix[m, 0]
        veh0y0 = yxmain(veh0x0)
        veh0v0 = sgenMatrix[m, 1]
        veh0a0 = sgenMatrix[m, 2]

        veh1x0 = sgenMatrix[m, 3]
        veh1y0 = yxmain(veh1x0)
        veh1v0 = sgenMatrix[m, 4]
        veh1a0 = sgenMatrix[m, 5]

        veh2x0 = sgenMatrix[m, 6]
        veh2y0 = yxmain(veh2x0)
        veh2v0 = sgenMatrix[m, 7]
        veh2a0 = sgenMatrix[m, 8]

        veh3x0 = sgenMatrix[m, 9]
        veh3y0 = yxtarget(veh3x0)
        veh3v0 = sgenMatrix[m, 10]
        veh3a0 = sgenMatrix[m, 11]

        veh4x0 = sgenMatrix[m, 12]
        veh4y0 = yxtarget(veh4x0)
        veh4v0 = sgenMatrix[m, 13]
        veh4a0 = sgenMatrix[m, 14]
        
        #候选终点范围
        minxf = min(veh0v0, veh4v0)*2
        maxxf = max(veh0v0, veh4v0) * 4
        xfsam = np.reshape(np.linspace(minxf, maxxf, NX), (-1, 1))    
        #候选变量
        xfcand = 0
        tfcand = 0
        fvalcand = 10000000000000000
        xcand = np.zeros((3*NT))
        alphacand = np.zeros((1, 4))
        feasiflag = 0
        ncand = 0
        errorcand = np.zeros((1, 3*NT))
        #对每个候选终点
        for n in range(NX):       
            xf = xfsam[n]
            #生成路径
            avector = path_sample(xf)
            #通过线性拟合方法将非线性防碰撞约束转换为线性约束
            ttmax, ss0, ssf, sslb, ssub, bbetaxubub, bbetaxublb, bbetaxlbub, bbetaxlblb, aalphaxubub, aalphaxublb, aalphaxlbub, aalphaxlblb, eerrorxubub, eerrorxublb, eerrorxlbub, eerrorxlblb = linear_fitting(avector, xf,  veh0v0,  veh3v0)
            #建立二次规划模，返回参数矩阵
            Aeq, beq, A, b, H, f = qp_foundation(ttmax, ss0, xf, ssf, sslb, ssub, veh0x0, veh0v0, veh0a0, veh1x0, veh1v0, veh1a0, veh2x0, veh2v0, veh2a0, veh3x0, veh3v0, veh3a0, veh4x0, veh4v0, veh4a0, bbetaxubub, bbetaxublb, bbetaxlbub, bbetaxlblb, aalphaxubub, aalphaxublb, aalphaxlbub, aalphaxlblb, eerrorxubub, eerrorxublb, eerrorxlbub, eerrorxlblb)  
        
            #********调用cvxopt建立二次规划模型
            '''
            min1/2xTHx+fTx
            s.t.Ax≤b
            Aeqx=beq 

            cvxopt.solvers.qp(P,q,G,h,A,b)
            '''
            H = matrix(H)
            f = matrix(f)
            A = matrix(A)
            b = matrix(b)
            Aeq = matrix(Aeq)
            beq = matrix(beq)
            results = solvers.qp(P=H, q=f, G=A, h=b, A=Aeq, b=beq)
            # print(results['primal objective'])
            #修改最大迭代次数100？
            '''
            x      *
            y
            s
            z
            status
            gap
            relative gap
            primal objective  *
            dual objective
            primal infeasibility
            dual infeasibility
            primal slack
            dual slack
            iterations
            '''            
            
            #********输出与存储计算结果
            if results['status'] == 'optimal':
                feasiflag = 1
                ReportMatrix[m, 1] = feasiflag
                ReportMatrix[m, 3*NT+6+n] = xf
                ReportMatrix[m, 3*NT+6+NX+n] = ttmax
                ReportMatrix[m, 3*NT+6+2*NX+n] = results['primal objective']
                # ReportMatrix[m, 4*NT+3+4+3*NX+n] = cputime
                # ReportMatrix[m, 4*NT+3+4+4*NX+n] = feasiflag   #可以记录是否是最优
                if results['primal objective'] < fvalcand:
                    xfcand = xf
                    tfcand = ttmax
                    fvalcand = results['primal objective']
                    ncand = n  #候选终点id
                    xcand = results['x']
                    alphacand = avector.T
                    # errorcand = [eerrorxubub.T, eerrorxublb.T, eerrorxlbub.T, eerrorxlblb.T]
        ReportMatrix[m, 2] = xfcand
        ReportMatrix[m, 3] = tfcand
        ReportMatrix[m, 4] = fvalcand
        ReportMatrix[m, 5] = ncand
        xcand  = np.array(xcand).reshape(-1)
        ReportMatrix[m, 6:3*NT+6] = xcand
        ReportMatrix[m, 4*NT+3+6:4*NT+3+10] = alphacand 
        # EnvCarMatrix[m, :N] = 
        # EnvCarMatrix[m, :NT] = veh1xs
        # EnvCarMatrix[m, NT:2*NT] = veh2xs
        # EnvCarMatrix[m, 2*NT:3*NT] = veh3xs
        # EnvCarMatrix[m, 3*NT:4*NT] = veh4xs
        # ReportError[m, :] = errorcand
    
    return 1
    
#依据候选终点位置，生成路径
def path_sample(xf_n):      #Frenet坐标系 s是车身坐标系的x
    x0 = 0   
    y0 = 0
    xf = xf_n
    yf = 3.75
    c0 = 0   #切线斜率：增加1单位x，y变化值为0
    cf = 0
    k0 = 0  #斜率的斜率：增加1单位x，斜率的变化值为0
    kf = 0
    
    avec = np.zeros((4, 1))
    gvec = np.zeros((4, 1))
    
    
    #先采用三次曲线
    for i in range(10):
        #路径曲线采用五次多项式；已知起点和终点XY坐标时，可分别带入原式、一阶导公式、二阶导公式进行求解
        gvec[0] = y0 - avec[0] + avec[1]*x0 + avec[2]*x0**2 + avec[3]*x0**3 
        gvec[1] = yf - avec[0] + avec[1]*xf + avec[2]*xf**2 + avec[3]*xf**3 
        gvec[2] = c0 - avec[1] + 2*avec[2]*x0 + 3*avec[3]*x0**2 
        gvec[3] = cf - avec[1] + 2*avec[2]*xf + 3*avec[3]*xf**2
        
        #g(a)是以路径曲线函数中系数为自变量的函数，一下分别对4个系数avec求导
        dg1da1 = 1
        dg1da2 = x0
        dg1da3 = x0**2
        dg1da4 = x0**3

        dg2da1 = 1
        dg2da2 = xf
        dg2da3 = xf**2
        dg2da4 = xf**3
        
        dg3da1 = 0
        dg3da2 = 1
        dg3da3 = 2*x0
        dg3da4 = 3*x0**2
        
        dg4da1 = 0
        dg4da2 = 1
        dg4da3 = 2*xf
        dg4da4 = 3*xf**2
        
        #存储导函数在np.array中
        Jg = np.zeros((4,4))
        
        Jg[0, 0] = dg1da1
        Jg[0, 1] = dg1da2
        Jg[0, 2] = dg1da3
        Jg[0, 3] = dg1da4
        
        Jg[1, 0] = dg2da1
        Jg[1, 1] = dg2da2
        Jg[1, 2] = dg2da3
        Jg[1, 3] = dg2da4
        
        Jg[2, 0] = dg3da1
        Jg[2, 1] = dg3da2
        Jg[2, 2] = dg3da3
        Jg[2, 3] = dg3da4

        Jg[3, 0] = dg4da1
        Jg[3, 1] = dg4da2
        Jg[3, 2] = dg4da3
        Jg[3, 3] = dg4da4
        
        avec = avec - np.dot(np.linalg.inv(Jg), gvec)   #np.linalg.inv(Jg): 逆矩阵                
    
    #计算获得五次多项式的系数后，定义曲线函数f(x)及导函数
    yxfun = lambda x: avec[0] + avec[1]*x + avec[2]*x**2 + avec[3]*x**3 
    dydxfun = lambda x: avec[1] + 2*avec[2]*x + 3*avec[3]*x**2
    
    return avec

#计算四个边界点和四个顶点   
def spatial_transformation(k, xsam, xf, avec):
    length = 2.4
    width = 0.9
    
    yxfun = lambda x: avec[0] + avec[1]*x + avec[2]*x**2 + avec[3]*x**3
    yxtarget = lambda x: 3.75   #目标车道中心线y值
    dydxtarget  =lambda x:0
    
    x0 = 0
    y0 = 0
    alpha = 0    #航向角？
    if xsam[k] <= xf:   #如果该样本点仍属于换道过程的路径范围
        x0 = xsam[k]
        y0 = yxfun(x0)
        alpha = np.arctan(x0)
    else:
        x0 = xsam[k]
        y0 = yxtarget(x0)
        alpha = dydxtarget(x0)
        
    e0 = np.cos(alpha) * length
    e1 = np.sin(alpha) * length
    e2 = np.cos(alpha) * width
    e3 = np.sin(alpha) * width
    e4 = e2
    e5 = e3
    e6 = e1
    e7= e0
    e8 = e2
    e9 = e3
    e10 = e2
    e11 = e3
    
    #车辆的四个顶点1、2、3、4  顶点坐标
    x1 = x0 + e0 + e3
    y1 = y0 + e1 - e2
    
    x2 = x0 + e0 - e5
    y2 = y0 + e1 + e4
    
    x3 = x0 - e7 - e11
    y3 = y0 - e6 + e10
    
    x4 = x0 - e7 + e9
    y4 = y0 - e6 - e8
    
    xvec = np.array([x1, x2, x3, x4])
    yvec = np.array([y1, y2, y3, y4])
    
    #(xl1, yl1) 和 (xl2, yl2)是车道线上的两个点（车道之间的分界线）
    xl1 = 0
    yl1 = 3.75/2
    xl2 = 25
    yl2 = 3.75/2
    
    #4个顶点组成了4条线段（2、1）（4、1）（3、4）（3、2），
    #以下分别检验这四条线段和车道线(xl1, yl1) 和 (xl2, yl2)的相交情况

    #交点公式：已知两条直线的四个点，求交点坐标
    xcpfun = lambda px1,py1,px2,py2,px3,py3,px4,py4: ((px1*py2 - py1*px2) * (px3 - px4) - (px1 - px2)*(px3*py4 - py3*px4)) / ((px1 - px2) * (py3 - py4) - (py1 - py2) * (px3 - px4) + 1e-8)
    ycpfun = lambda px1,px2,px3,px4,py1,py2,py3,py4: ((px1*py2 - py1*px2) * (py3 - py4) - (py1 - py2)*(px3*py4 - py3*px4)) / ((px1 - px2) * (py3 - py4) - (py1 - py2) * (px3 - px4) + 1e-8)
    
    #（2、1）和车道线交点
    px = xcpfun(x2, y2, x1, y1, xl1, yl1, xl2, yl2)
    py = ycpfun(x2, y2, x1, y1, xl1, yl1, xl2, yl2)
    xmax = max(x2, x1)
    xmin = min(x2, x1)
    candx1 = -10000
    candy1 = -10000
    #如果交点在线段上，认为是有效的
    if px <= xmax and px >= xmin:
        candx1 = px
        candy1 = py
    
    #（4、1）和车道线交点
    px = xcpfun(x4, y4, x1, y1, xl1, yl1, xl2, yl2)
    py = ycpfun(x4, y4, x1, y1, xl1, yl1, xl2, yl2)
    xmax = max(x4, x1)
    xmin = min(x4, x1)
    candx2 = -10000
    candy2 = -10000
    #如果交点在线段上，认为是有效的
    if px <= xmax and px >= xmin:
        candx2 = px
        candy2 = py
        
    #（3、4）和车道线交点
    px = xcpfun(x3, y3, x4, y4, xl1, yl1, xl2, yl2)
    py = ycpfun(x3, y3, x4, y4, xl1, yl1, xl2, yl2)
    xmax = max(x3, x4)
    xmin = min(x3, x4)
    candx3 = -10000
    candy3 = -10000
    #如果交点在线段上，认为是有效的
    if px <= xmax and px >= xmin:
        candx3 = px
        candy3 = py

    #（3、2）和车道线交点
    px = xcpfun(x3, y3, x2, y2, xl1, yl1, xl2, yl2)
    py = ycpfun(x3, y3, x2, y2, xl1, yl1, xl2, yl2)
    xmax = max(x3, x2)
    xmin = min(x3, x2)
    candx4 = -10000
    candy4 = -10000
    #如果交点在线段上，认为是有效的
    if px <= xmax and px >= xmin:
        candx4 = px
        candy4 = py
        
    # 开始找两个交点，并寻找两个车道分别最靠前和最靠后的点（共四个点）
    candxmax = max([candx1, candx2, candx3, candx4])
    if candxmax > -10000:   #当车身与车道线有交点
        xlbub = -10000       #前面的ub代表目标车道、lb代表原车道；后面的ub代表前侧、lb代表后侧
        xubub = -10000
        xvecmax = candxmax  
        for i in range(4):
            if xvec[i] >= xvecmax:    #交点x坐标最大也不会超过顶点
                if yvec[i] <= yl1:    #顶点y小于交点y,说明该顶点在原车道
                    xlbub = xvec[i]   #两个车道分别最前侧的点依此确定
                    xubub = candxmax
                else:
                    xlbub = candxmax
                    xubub = xvec[i]
                xvecmax = xvec[i]   #更新
        #四个假设交点里只可能有0、1、2个交点三种情况
        #将-10000转为10000，才能使用min()
        candxmin = min([littleone + 20000 if  littleone == -10000 else littleone for littleone in [candx1,candx2,candx3,candx4]]) 
        xlblb = -10000 
        xublb = -10000 
        xvecmin = candxmin 
        for i in range(4):           
            if xvec[i] <= xvecmin :   
                if yvec[i] <= yl1 :         #两个车道分别最后侧的点依此确定
                    xlblb = xvec[i] 
                    xublb = candxmin 
                else:
                    xlblb = candxmin 
                    xublb = xvec[i] 
                xvecmin = xvec[i] 
    else:    #没有交点的情况，即车辆未压线
        if max(yvec) <= yl1:   #在原车道
            xubub = x2       #取最靠近目标车道的2号顶点作为前后侧依据
            xublb = x2       
            xlbub = max(xvec) 
            xlblb = min(xvec) 
        else:                 #在目标车道
            xlbub = x4       #取最靠近原车道的4号顶点作为前后侧依据
            xlblb = x4 
            xubub = max(xvec) 
            xublb = min(xvec) 
    #防碰撞约束只需判断x坐标，所以只计算四个x
    
    return x0, y0, x1, x2, x3, x4, xlbub, xlblb, xubub, xublb

#通过线性拟合方法将非线性防碰撞约束转换为线性约束
def linear_fitting(avec, xf, veh0v0,  veh4v0 ):
    # xf = 25 
    # NT = 20 

    ff = lambda x: (1+(avec[1]+2*avec[2]*x+3*avec[3]*x**2)**2)**0.5
    s0 = 0 
    sf = integrate.quad(ff,0,xf)[0]   #求定积分，获得终点时换道路径总长度。PS：这里sf不包括换道后跟驰距离！
    
    tmax = sf / veh0v0    #总时间：换道+换道后跟驰

    sef = tmax * veh4v0  #总驾驶距离:换道+换道后跟驰

    xef = xf + (sef-sf)     #总纵向距离:换道+换道后跟驰

    xsam = np.reshape(np.linspace(0, xef, NS*NT+1), (-1, 1))   #用来拟合的样本点的x位置，column vector(241,1)
    suvec = np.zeros((xsam.shape[0],1))   #(241,1)colunm vector。  241个样本点
    #求每个样本点 从起点开始的曲线距离s
    for k in range(xsam.shape[0]):    
        if xsam[k] <= xf : 
            suvec[k] = integrate.quad(ff,xsam[0],xsam[k])[0] 
        else:
            suvec[k] = sf + (xsam[k] - xf)  #因为是直线了所以不用求导
    
    xububvec = np.zeros((xsam.shape[0],1))       #四个车辆边界点的x
    xublbvec = np.zeros((xsam.shape[0],1)) 
    xlbubvec = np.zeros((xsam.shape[0],1)) 
    xlblbvec = np.zeros((xsam.shape[0],1)) 
    x0vec = np.zeros((xsam.shape[0],1))          #0是中心点，1、2、3、4分别对应egoCar的四个顶点
    y0vec = np.zeros((xsam.shape[0],1)) 
    x1vec = np.zeros((xsam.shape[0],1)) 
    x2vec = np.zeros((xsam.shape[0],1)) 
    x3vec = np.zeros((xsam.shape[0],1)) 
    x4vec = np.zeros((xsam.shape[0],1)) 

    for k in range(xsam.shape[0]):   #对于每个样本点   (NS*NT+1)=（12*20+1）
        #计算四个边界点和四个顶点   
        x0, y0, x1, x2, x3, x4, xlbub, xlblb, xubub, xublb = spatial_transformation(k, xsam, xf, avec)     
        x0vec[k] = x0 
        y0vec[k] = y0 
        xububvec[k] = xubub 
        xublbvec[k] = xublb 
        xlbubvec[k] = xlbub 
        xlblbvec[k] = xlblb 
        x1vec[k] = x1  # x position for conner 1
        x2vec[k] = x2  # x position for conner 2
        x3vec[k] = x3  # x positoin for conner 3
        x4vec[k] = x4  # x position for conner 4

    # NT =拟合次数=I个时间段      
    alphaxlbub = np.zeros((NT,1)) 
    alphaxlblb = np.zeros((NT,1)) 
    alphaxubub = np.zeros((NT,1)) 
    alphaxublb = np.zeros((NT,1)) 
    betaxlbub = np.zeros((NT,1)) 
    betaxlblb = np.zeros((NT,1)) 
    betaxubub = np.zeros((NT,1)) 
    betaxublb = np.zeros((NT,1)) 
    errorxlbub = np.zeros((NT,1)) 
    errorxlblb = np.zeros((NT,1)) 
    errorxubub = np.zeros((NT,1)) 
    errorxublb = np.zeros((NT,1)) 
    sub = np.zeros((NT,1)) 
    slb = np.zeros((NT,1)) 
    
    for i in range(NT):            
        #采样范围的上下限。 目标车道前车速度(veh4v0)>ego车速度(veh0v0)，作为计算依据如下：
        temslb = 0.8*(i*tmax/NT)*veh0v0  
        temsub = (i*tmax/NT)*veh4v0 
        #查找离上下限位置最近的样本点
        findslb = np.where(abs(suvec - temslb) == min(abs(suvec - temslb)))   #findslb(1)获得列向量abs(suvec - temslb)中满足条件的索引
        findsub = np.where(abs(suvec - temsub) == min(abs(suvec - temsub))) 
        LI = findslb[0][0]  #下限索引，靠近起点的一侧
        HI = findsub[0][0]  #exsample:  (array([179]), array([0])) --> 179
        
        if i < (NT-1):
            slb[i] = suvec[LI]   #拟合范围的下限是某一样本点的位置
        else:
            slb[i] = sf        #最后一个时间段，拟合范围的下限就是代表换道结束的Sf    
        #Q5：怎么保证最后一段一定处于跟驰阶段（suvec(LI)<sf）或只有最后一段包括跟驰阶段？老师说有舍有得
        sub[i] = suvec[HI]
        
        #线性拟合方法：高中数学线性回归方程
        #知识补充：统计会将理论与实际间的差别表示出来，也就是“误差”。残差是真实值和预测值间的差值。
        #普通最小二乘法给出的判断标准是：残差平方和（SSE）的值达到最小。
        
        # mdlxubub = fitlm(suvec(LI:HI)',xububvec(LI:HI)) 
        #注意：matlab中切片操作例如[2:4]包括第2和第4个信息，python不包括4，所以HI+1
        mx = np.mean(suvec[LI:HI+1])     #拟合范围中的样本点x平均值
        my = np.mean(xububvec[LI:HI+1])   #拟合范围中的样本点，各自的目标车道前侧顶点坐标x平均值
        beta = np.sum((suvec[LI:HI+1]-mx)*(xububvec[LI:HI+1]-my)) / (np.sum((suvec[LI:HI+1]-mx)**2)+ 1e-8)  #防止分母为0
        alpha = my - beta*mx    
        py = alpha + beta*suvec[LI:HI+1] 
        errorxubub[i] = max(xububvec[LI:HI+1] - py)    
        alphaxubub[i] = alpha 
        betaxubub[i] = beta         
        
        # mdlxublb = fitlm(suvec(LI:HI)',xublbvec(LI:HI)) 
        mx = np.mean(suvec[LI:HI+1]) 
        my = np.mean(xublbvec[LI:HI+1]) 
        beta = np.sum((suvec[LI:HI+1]-mx)*(xublbvec[LI:HI+1]-my)) / (np.sum((suvec[LI:HI+1]-mx)**2) + 1e-8) 
        alpha = my - beta*mx 
        py = alpha + beta*suvec[LI:HI+1] 
        errorxublb[i] = min(xublbvec[LI:HI+1]-py)                                                       
        alphaxublb[i] = alpha 
        betaxublb[i] = beta 
        #Q6：前车+误差项，后车-误差项？这里用min是因为误差是负数吗，为什么？
        #AS：第一问猜测，为了更保险，选择可能的最靠后和最靠前的位置。
        
        # mdlxlbub = fitlm(suvec(LI:HI)',xlbubvec(LI:HI)) 
        mx = np.mean(suvec[LI:HI+1]) 
        my = np.mean(xlbubvec[LI:HI+1]) 
        beta = np.sum((suvec[LI:HI+1]-mx)*(xlbubvec[LI:HI+1]-my)) / (np.sum((suvec[LI:HI+1]-mx)**2)+ 1e-8)  
        alpha = my - beta*mx   
        py = alpha + beta*suvec[LI:HI+1] 
        errorxlbub[i] = max(xlbubvec[LI:HI+1] - py) 
        alphaxlbub[i] = alpha 
        betaxlbub[i] = beta 
        
        # mdlxlblb = fitlm(suvec(LI:HI)',xlblbvec(LI:HI)) 
        mx = np.mean(suvec[LI:HI+1]) 
        my = np.mean(xlblbvec[LI:HI+1]) 
        beta = np.sum((suvec[LI:HI+1]-mx)*(xlblbvec[LI:HI+1]-my)) / (np.sum((suvec[LI:HI+1]-mx)**2) + 1e-8) 
        alpha = my - beta*mx 
        py = alpha + beta*suvec[LI:HI+1] 
        errorxlblb[i] = min(xlblbvec[LI:HI+1]-py) 
        alphaxlblb[i] = alpha 
        betaxlblb[i] = beta 
    
    return tmax, s0, sf, slb, sub, betaxubub, betaxublb, betaxlbub, betaxlblb, alphaxubub, alphaxublb, alphaxlbub, alphaxlblb, errorxubub, errorxublb, errorxlbub, errorxlblb

#建立二次规划模型
def qp_foundation(tmax, s0, xf, sf, slb, sub, veh0x0, veh0v0, veh0a0, veh1x0, veh1v0, veh1a0, veh2x0, veh2v0, veh2a0, veh3x0, veh3v0, veh3a0, veh4x0, veh4v0, veh4a0, betaxubub, betaxublb, betaxlbub, betaxlblb, alphaxubub, alphaxublb, alphaxlbub, alphaxlblb, errorxubub, errorxublb, errorxlbub, errorxlblb):
    '''
    x = 
    s: 1 to NT
    speed: NT+1 to 2*NT
    acc: 2*NT+1 to 3*NT
    '''

    #------------变量初始化
    w1 = 0.0343  #跟驰模型经验参数
    w2 = 0.948
    dxstar = 30
    tf = tmax    #由外生参数计算获得的换道总时间
    dt = tf/NT   #用于线性化的单个时间区间长度

    Aeq = np.zeros((2*NT,3 * NT))      #2*NT个等式约束
    beq = np.zeros((2*NT, 1))
    A = np.zeros((4*NT+NT-1,3*NT))    #5*NT-1个不等式约束
    b = np.zeros((4*NT+NT-1, 1))    
    
    line = 0
    
    #------------等式约束
    # definition for speed
    for i in range(NT):
        if i == 0:
            # x(NT+i) = (x(i)-s0)/dt               
            # 1*x(NT+i)+(-1/dt)* x(i) = -s0/dt                   
            Aeq[line,NT+i] = 1
            Aeq[line,i] = -1/dt                                 
            beq[line] = -s0/dt
            line = line + 1
        else:                      
            # x(NT+i) = (x(i)-x(i-1))/dt 
            # x(NT+i) + (-1/dt)*x(i) + (1/dt)*x(i-1) = 0 
            Aeq[line,NT+i] = 1
            Aeq[line,i] = -1/dt
            Aeq[line,i-1] = 1/dt
            beq[line] = 0
            line = line + 1
    
    # definition for acc
    for i in range(NT):
        if i == 0:
            # x(2*NT+i) = (x(NT+i) - veh0v0)/dt 
            # x(2*NT+i) + (-1/dt)*x(NT+i) = -veh0v0/dt 
            Aeq[line,2*NT+i] = 1
            Aeq[line,NT+i] = -1/dt
            beq[line] = -veh0v0/dt 
            line = line + 1
        else: 
            # x(2*NT+i) = (x(NT+i) - x(NT+i-1))/dt 
            # x(2*NT+i) + (-1/dt)*x(NT+i) + (1/dt)*(NT+i-1) = 0 
            Aeq[line,2*NT+i] = 1 
            Aeq[line,NT+i] = -1/dt 
            Aeq[line,NT+i-1] = 1/dt 
            beq[line] = 0 
            line = line + 1 
    
    # ------------不等式约束1：collision avoid         
    line = 0 
    for i in range(NT):                 #没有车辆长度的部分？？？
        # xubub
        # veh0x0+alphaxubub(i)+betaxubub(i)*x(i)+ errorxubub(i) < veh4x0 + veh4v0*i*dt+0.5*veh4a0*(i*dt)**2 
        # betaxubub(i)*x(i) < veh4x0 + veh4v0*i*dt+0.5*veh4a0*(i*dt)**2 - veh0x0 - alphaxubub(i) - errorxubub(i) 
        A[line,i] = betaxubub[i] 
        b[line] = veh4x0 + veh4v0*i*dt + 0.5*veh4a0*(i*dt)**2 - veh0x0 - alphaxubub[i] - errorxubub[i] 
        line = line + 1 

        # xublb
        # veh3x0 + veh3v0*i*dt + 0.5*veh3a0*(i*dt)**2 + (1/6)*x(4*NT+1)*(i*dt)**3 < veh0x0 + alphaxublb(i) + betaxublb(i)*x(i) + errorxublb(i) 
        # (1/6)*(i*dt)**3*x(4*NT+1) + (-betaxublb(i))*x(i) < veh0x0 + alphaxublb(i) + errorxublb(i) - veh3x0 - veh3v0*i*dt - 0.5*veh3a0*(i*dt)**2 
        A[line,i] = -betaxublb[i] 
        b[line] = veh0x0 + alphaxublb[i] + errorxublb[i] - veh3x0 - veh3v0*i*dt - 0.5*veh3a0*(i*dt)**2 
        line = line + 1    

        # xlbub
        # veh0x0 + alphaxlbub(i) + betaxlbub(i)*x(i) + errorxlbub(i) < veh2x0 + veh2v0*i*dt+0.5*veh2a0*(i*dt)**2 
        # betaxlbub(i)*x(i) < veh2x0 + veh2v0*i*dt+0.5*veh2a0*(i*dt)**2 - veh0x0 - alphaxlbub(i) - errorxlbub(i) 
        A[line,i] = betaxlbub[i] 
        b[line] = veh2x0 + veh2v0*i*dt+0.5*veh2a0*(i*dt)**2 - veh0x0 - alphaxlbub[i] - errorxlbub[i] 
        line = line + 1 
        
        #xlblb
        #veh1x0 + veh1v0*I*dt + 0.5*veh1a0*(i*dt)**2  < veh0x0 + alphaxlblb(i) + betaxlblb(i)*x(i) + errorxlblb(i)
        #-betaxlblb(i)*x(i) < veh0x0 + alphaxlblb(i) + errorxlblb(i) - veh1x0 - veh1v0*i*dt - 0.5*veh1a0*(i*dt)**2 
        A[line,i] = -betaxlblb[i] 
        b[line] = veh0x0 + alphaxlblb[i] + errorxlblb[i] - veh1x0 - veh1v0*i*dt - 0.5*veh1a0*(i*dt)**2 
        line = line + 1    
            
    #------------不等式约束2：后车不能超过前车
    for i in range(1, NT):
        # x(i-1) < x(i)
        # x(i-1) + (-1)*x(i) < 0
        A[line,i-1] = 1 
        A[line,i] = -1 
        b[line] = 0 
        line = line + 1 


    #------------各自由变量的取值范围
    lb = np.zeros((3*NT,1))
    ub = np.zeros((3*NT,1))
    # curve distance
    lb[0:NT] = slb[0:NT] 
    ub[0:NT] = sub[0:NT] 
    # speed
    lb[NT:2*NT] = 0.01 
    ub[NT:2*NT] = 30 
    # acc
    lb[2*NT:3*NT] = -4 
    ub[2*NT:3*NT] = 4 
    
    #------------目标函数中二次项的系数矩阵
    H = 0*np.eye(3*NT)   
    # H[NT-1,NT-1] = -1/((5*tf)**2) #换道曲线总长度，为什么是负的？
    for i in range(2*NT, 3*NT):
        H[i,i] = 1/NT                           #每个时间区间内的加速度
    
    #目标函数中一次项的系数矩阵
    f = np.zeros((3*NT, 1))   
    
    return Aeq, beq, A, b, H, f

#优化结果可视化
def results_plot(ReportM, EnvCarM): 
    path_avec = ReportM[0, 4*NT+3+6:4*NT+3+10]    #print(path_avec.shape)  => (4,)
    y0 = 0
    y1 = 3.75
    x = np.linspace(0, 100, 100)
    print(path_avec)
    y = path_avec[0] + path_avec[1]*x + path_avec[2]*x**2 + path_avec[3]*x**3 
    # plt.figure(figsize=(6,4)) # 设置图片大小
    plt.ylim(y0, y1)
    plt.xlim(0,100)
    plt.plot(x, y)
    # plt.legend()
    plt.show()
    
    return 1

if __name__ == '__main__':                                    
    #全局变量
    NumGen = 1
    NT = 20   #时间区间数
    NS = 12   #每个时间区间内离散样本点数
    NX =20  #终点候选点个数
    ReportMatrix = np.zeros((NumGen, 4*NT+3+10+5*NX))  #输出文件
    # ReportError = np.zeros((NumGen, 4*NT))
    EnvCarMatrix = np.zeros((NumGen, 3*NT+4*NT+1))     #拟合公式三个参数 周围四辆车的x位置
    
    
    #生成场景
    sgenMatrix = scenario_generation()
    # sgenMatrix.tofile('sgen1.csv', sep = ',')
    
    #执行优化模型
    qp_execution(sgenMatrix)
    
    # results_plot(ReportMatrix, EnvCarMatrix)
    
    # np.savetxt('results0921.csv', ReportMatrix, delimiter=',')