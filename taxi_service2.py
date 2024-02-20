import numpy as np
import random
import matplotlib.pyplot as plt

def probability_fun(j,i,k, h_mtx, w_arr,para):
 temp = sum(np.exp(-para*(h_mtx[j][m] + w_arr[k, m])) for m in range(4)) + 1e-8
 return np.exp(-para*(h_mtx[j][i]+w_arr[k, i]))/temp

def probability_fun2(j,i,k, h_mtx, w_arr, para):
 temp = sum(np.exp(-para*(h_mtx[j][m] + w_arr[k, m])) for m in range(4)) + 1e-8
 return np.exp(-para*h_mtx[j][i])/temp



def fixed_point_alg(para_c):
    max_iterations = 100
    convergence_threshold = 0.0001
    num_nodes = 4
    num_vehicles = 300


    travel_time_mtx = [[0, 0.25, 0.3, 0.35], 
                    [0.25, 0, 0.3, 0.2],
                    [0.3, 0.3, 0, 0.25],
                    [0.35, 0.2, 0.25, 0]]

    od_demand_mtx = [[0, 50, 20, 20],
                    [40, 0, 15, 25],
                    [20, 10, 0, 50],
                    [10, 20, 30, 0]]

    demand_d_ls = []
    demand_o_ls = []
    for j in range(num_nodes):
        demand_d_ls.append(sum(od_demand_mtx[i][j] for i in range(num_nodes)) )  #以j作为终点的出行需求量
    for i in range(num_nodes):
        demand_o_ls.append(sum(od_demand_mtx[i][j] for j in range(num_nodes)) ) #以i作为起点的出行需求量


    waittime_arr = np.zeros((max_iterations, num_nodes))    #单位hour
    taxi_movements_mtx = np.zeros((num_nodes, num_nodes))

    # #initialize the waittime in k = 0
    # for i in range(4):  
    #     waittime_arr[0,i] = random.randint(0, 100)/100 
    waittime_arr[0,0] = 0.03
    waittime_arr[0,1] = 0.03
    waittime_arr[0,2] = 0.03
    waittime_arr[0,3] = 0.03

    for iteration in range(max_iterations - 1):
        #step 1 计算空载车辆行驶次数矩阵
        for j in range(num_nodes):
            for i in range(num_nodes):
                probability_val = probability_fun(j, i, iteration, travel_time_mtx, waittime_arr, para_c)
                taxi_movements_mtx[j, i] = demand_d_ls[j]*probability_val

        #step 2 更新出租车等待时间
        for i in range(num_nodes):
            waittime_arr[iteration+1, i] = -para_c*np.log(demand_o_ls[j]) + para_c*np.log(sum(demand_d_ls[jj] * probability_fun2(jj, i, iteration, travel_time_mtx, waittime_arr, para_c) for jj in range(num_nodes)))
        
        
        # step 3 判断是否达到收敛条件
        if all(abs(waittime_arr[iteration+1, i] - waittime_arr[iteration, i]) < convergence_threshold for i in range(num_nodes)):
            print("在最大迭代次数内收敛！")
            break

    #输出结果
    # print(waittime_arr)
    return waittime_arr,iteration+1



def plot1(data):
    # 设置图形大小
    plt.figure(figsize=(8, 6))

    # 绘制折线图
    num_nodes = len(data[0])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 颜色列表
    linestyles = ['-', '--', '-.', ':']  # 线条样式列表
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h']  # 标记样式列表
    for node in range(num_nodes):
        node_data = [row[node] for row in data]
        plt.plot(range(1, len(data) + 1), node_data, label=f'Node {node+1}', color=colors[node % len(colors)], linestyle=linestyles[node % len(linestyles)], marker=markers[node % len(markers)])

    # 添加标题和标签
    plt.title('Wait Time vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Wait Time (hour)')

    # 调整坐标轴刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 添加图例，并将其放置在左上方
    plt.legend(loc='upper right', fontsize=12)

    # 将横坐标值改为整数
    plt.xticks(range(1, len(data) + 1))

    # 保存为矢量图格式，适用于学术论文
    plt.savefig('wait_time_vs_iteration_plot.svg', format='svg')

    # 显示图形
    plt.show()


def plot2(data):
    # 设置图形大小
    plt.figure(figsize=(8, 6))

    # 绘制折线图
    num_nodes = len(data[0])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 颜色列表
    linestyles = ['-', '--', '-.', ':']  # 线条样式列表
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h']  # 标记样式列表
    for node in range(num_nodes):
        node_data = [row[node] for row in data]
        plt.plot(np.arange(0.1, 1.0, 0.1), node_data, label=f'Node {node+1}', color=colors[node % len(colors)], linestyle=linestyles[node % len(linestyles)], marker=markers[node % len(markers)])
    
    # 添加标题和标签
    plt.title(r'Wait Time vs. Parameter $\theta$')
    plt.xlabel(r'Parameter $\theta$')
    plt.ylabel('Wait Time (hour)')

    # 调整坐标轴刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 添加图例，并将其放置在左上方
    plt.legend(loc='upper left', fontsize=12)

    # 保存为矢量图格式，适用于学术论文
    plt.savefig('wait_time_plot.svg', format='svg')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    #最终结果存储
    final_res_iter = []
    final_res_para = []

    # #可视化各节点等待时间的迭代过程
    # res, conv_idx = fixed_point_alg(para_c=0.1)
    # for idx in range(conv_idx+1):
    #     final_res_iter.append(res[idx].tolist())

    # print(final_res_iter)
    # plot1(final_res_iter)


    #探索不同sita对等待时间的影响
    for hyper_para in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        res, conv_idx = fixed_point_alg(hyper_para)
        final_res_para.append(res[conv_idx].tolist())
    print(final_res_para)

    plot2(final_res_para)
