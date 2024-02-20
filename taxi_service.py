import random
import networkx as nx
import matplotlib.pyplot as plt

def flow_allocation():
    # 定义节点数量
    num_nodes = 4

    # 创建一个空的有向图
    G = nx.DiGraph()

    # 设置随机数生成器的种子
    random.seed(42)

    # 生成节点之间的随机旅行时间和链接编号
    link_id = 1
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            travel_time = random.randint(1, 10)  # 将1和10替换为你想要的范围
            G.add_edge(i, j, travel_time=travel_time, link_id=link_id)
            G.add_edge(j, i, travel_time=travel_time, link_id=link_id)
            link_id += 1

    # 初始化节点的流量值
    node_flows = [0] * num_nodes

    # 迭代更新节点的流量值
    max_iterations = 100
    convergence_threshold = 0.001
    for iteration in range(max_iterations):
        prev_node_flows = node_flows.copy()

        for node in range(num_nodes):
            inflow = sum(prev_node_flows[i] for i in G.predecessors(node))
            outflow = sum(prev_node_flows[i] for i in G.successors(node))
            node_flows[node] = inflow - outflow

        # 判断是否达到收敛条件
        if all(abs(node_flows[i] - prev_node_flows[i]) < convergence_threshold for i in range(num_nodes)):
            break

    # 打印最终的节点流量值
    for node, flow in enumerate(node_flows):
        print(f"Node {node}: Flow = {flow}")


def get_demand():
    # 设置随机数生成器的种子
    random.seed(42)
    # Define the number of nodes in the network
    num_nodes = 4

    # Create an empty adjacency matrix to represent the traveltime matrix
    od_mtx = [[0] * num_nodes for _ in range(num_nodes)]

    # Generate random travel times between nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            request_demand = random.randint(0, 100)  # Replace 1 and 10 with your desired range
            od_mtx[i][j] = request_demand
            od_mtx[j][i] = request_demand

    return od_mtx

def draw_network_index():
    # 定义节点数量
    num_nodes = 4

    # 创建一个空的有向图
    G = nx.DiGraph()

    # 设置随机数生成器的种子
    random.seed(42)

    # 生成节点之间的随机旅行时间和链接编号
    link_id = 1
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            travel_time = random.randint(1, 10) # 将1和10替换为你想要的范围
            G.add_edge(i, j, travel_time=travel_time, link_id=(link_id, link_id+1))
            G.add_edge(j, i, travel_time=travel_time, link_id=(link_id, link_id+1))
            link_id += 2

    # 绘制路网图
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, edge_color='gray', width=1.5, arrows=True)

    # 绘制边的旅行时间标签
    edge_labels = nx.get_edge_attributes(G, 'link_id')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # 显示图形
    plt.show()
    
    return 0 


if __name__ == '__main__':

    # flow_allocation()

    draw_network_index()
    # so how can i get the link_id in the networkx graph?
    # while()